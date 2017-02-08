import FWCore.ParameterSet.Config as cms
from copy import deepcopy

# this is a pure Python module to build the deeply nested PSets that describe a 
# SummationSpecification.
# The C++ code assumes the form is fully correct, so you should always use this, 
# which outputs a valid form.

# these need to stay in sync with the C++ enums.  TODO: Can we use Python3 enum or sth.?
NO_TYPE  = cms.int32(0)
GROUPBY  = cms.int32(1)  # only "SUM", real histograms
EXTEND_X = cms.int32(2)  # use geometry column as coordinate axis, concatenate 
EXTEND_Y = cms.int32(3)
COUNT    = cms.int32(4)  # drop all values, only count entries. Atm only step1.
REDUCE   = cms.int32(5)  # histogram-to-scalar operator for harvesting, atm only MEAN
SAVE     = cms.int32(6)  # atm not used in execution. Marks stage1/2 switch.
CUSTOM   = cms.int32(7)  # call callback in harvesting
USE_X    = cms.int32(8)  # use arg-th fill(...) parameter for the respective axis. 
USE_Y    = cms.int32(9)
USE_Z    = cms.int32(10)
PROFILE  = cms.int32(11) # marker for step1 to make a profile, related to REDUCE(MEAN)

NO_STAGE = cms.int32(0)
FIRST    = cms.int32(1)  # first grouping, before and/or after counting 
STAGE1   = cms.int32(2)  # USE/EXTEND/PROFILE for step1
STAGE2   = cms.int32(3)  # REDUCE/EXTEND/GROUPBY/CUSTOM for harvesting

def val(maybecms):
  if hasattr(maybecms, "value"):
    return maybecms.value()
  else:
    return maybecms

def parent(path):
  parts = val(path).split("/")
  return "/".join(parts[0:len(parts)-1])

# do not change values here, Pass in a PSet instead
DefaultConf = cms.PSet(enabled = cms.bool(True))


# The Specification format is very rigid and looks much less like a program in
# the internal form:
# - There is one entry FIRST, which is a GROUPBY(SUM) or COUNT and some columns
# - There is another entry FIRST, which is a GROUPBY(SUM) and some columns iff
#   the one before was COUNT
# - There are some entries STAGE1
#  - There is one entry per dimension (ordered)
#  - which is either USE_* or EXTEND_*
#  - with one column, that is usually NOT listed in first.
#  - There is optionally an entry PROFILE to make a profile.
# - There are 0-n steps STAGE2, which are one of GROUPBY(SUM), EXTEND_X, CUSTOM
#  - The argument for GROUPBY and EXTEND_X is a subset of columns of last step
#  - CUSTOM may have an arbitrary argument that is simply passed down
#  - SAVE is ignored

class Specification(cms.PSet):
  def __init__(self, conf = DefaultConf):
    super(Specification, self).__init__()
    self.spec = cms.VPSet()
    self.conf = conf
    self._activeColumns = set()
    self._state = FIRST

  def __deepcopy__(self, memo):
    # override deepcopy to not copy .conf: it should remain a reference
    # w/o this it is not cleanly possible to build a per-module switch.
    t = Specification(self.conf)
    t.spec = deepcopy(self.spec, memo)
    return t 

  def groupBy(self, cols, mode = "SUM"):
    cnames = val(cols).split("/")
    newstate = self._state

    if self._state == FIRST:
      cname = cnames
      if mode != "SUM":
        raise Exception("First grouping must be SUM") 
      if "Event" in cnames:
        cnames.remove("Event"); # per-Event grouping is done automatically
        t = COUNT
        mode =  "COUNT"
        newstate = FIRST
      else:
        t = GROUPBY
        newstate = STAGE1

    if self._state == STAGE1:
      cname = self._activeColumns.difference(cnames)
      if len(cname) != 1:
        raise Exception("EXTEND must drop exactly one column.")

      if mode == "EXTEND_X":
        self._x.type = EXTEND_X
        self._x.columns = cms.vstring(cname)
      elif mode == "EXTEND_Y":
        self._y.type = EXTEND_Y
        self._y.columns = cms.vstring(cname)
      else:
        raise Exception("Only EXTEND_X or EXTEND_Y allowed here, not " + mode)

      # remove the column in earlier steps, we always re-extract in step1.
      c = list(cname)[0]
      for s in self.spec:
        if s.stage == FIRST and c in s.columns:
          s.columns.remove(c)
      if c in self._activeColumns:
        self._activeColumns.remove(c)
      if c in self._lastColumns:
        self._lastColumns.remove(c)

      return self # done here, no new step to add

    if self._state == STAGE2:
      cname = cnames
      if self._activeColumns.issubset(cname):
        raise Exception("Harvesting GROUPBY must drop some columns")
      if mode == "EXTEND_X":
        t = EXTEND_X
      elif mode == "SUM":
        t = GROUPBY
      else:
        raise Exception("Currently only EXTEND_X and SUM supported in harvesting, not " + mode)

    self._activeColumns = set(cnames)
    self._lastColumns = cnames
    self._lastMode = mode

    self.spec.append(cms.PSet(
      type = t, 
      stage = self._state, 
      columns = cms.vstring(cname),
      arg = cms.string(mode)
    ))

    if newstate == STAGE1 and self._state == FIRST:
      # emit standard column assignments, will be changed later
      self._x = cms.PSet(
        type = USE_X, stage = STAGE1,
        columns = cms.vstring(),
        arg = cms.string("")
      )
      self.spec.append(self._x)
      self._y = cms.PSet(
        type = USE_Y, stage = STAGE1,
        columns = cms.vstring(),
        arg = cms.string("")
      )
      self.spec.append(self._y)
      self._z = cms.PSet(
        type = USE_Z, stage = STAGE1,
        columns = cms.vstring(),
        arg = cms.string("")
      )
      self.spec.append(self._z)

    self._state = newstate

    return self

  def reduce(self, sort):
    if self._state == FIRST:
      if sort != "COUNT":
        raise Exception("First statement must be groupBy.")
      self.spec[0].type = COUNT # this is actually a noop
      # groupBy already saw the "Event" column and set up counting.

      return self

    if self._state == STAGE1:
      if sort == "MEAN":
        self.spec.append(cms.PSet(
          type = PROFILE, stage = STAGE1,
          columns = cms.vstring(), arg = cms.string("")
        ))
      return self

    if sort != "MEAN":
      raise Exception("Harvesting allows only reduce(MEAN) at the moment, not " + sort)

    self.spec.append(cms.PSet(
      type = REDUCE, 
      stage = self._state, 
      columns = cms.vstring(),
      arg = cms.string(sort)
    ))
    return self

  def save(self):
    if self._state == FIRST:
      raise Exception("First statement must be groupBy.")

    if self._state == STAGE1:
      # end of STAGE1, fix the parameter assignments
      n = 1
      if self._x.type == USE_X: self._x.arg = cms.string(str(n)); n = n+1
      if self._y.type == USE_Y: self._y.arg = cms.string(str(n)); n = n+1
      if self._z.type == USE_Z: self._z.arg = cms.string(str(n)); n = n+1
      # we don't know how many parameters the user wants to pass here, but the 
      # HistogramManager knows. So we just add 3.

    # SAVE is implicit in step1 and ignored in harvesting, so not really needed.
    #self.spec.append(cms.PSet(
    #  type = SAVE, 
    #  stage = self._state, 
    #  columns = cms.vstring(),
    #  arg = cms.string("")
    #))
    self._state = STAGE2
    return self

  def custom(self, arg = ""):
    if self._state != STAGE2:
      raise Exception("Custom processing exists only in Harvesting.")
    self.spec.append(cms.PSet(
      type = CUSTOM, 
      stage = self._state, 
      columns = cms.vstring(),
      arg = cms.string(arg)
    ))
    return self


  def saveAll(self):
    self.save()
    columns = self._lastColumns
    for i in range(len(columns)-1, 0, -1):
      cols = columns[0:i]
      self.groupBy("/".join(cols), self._lastMode)
      self.save()
    return self

  # this is used for serialization, and for that this is just a PSet.
  def pythonTypeName(self):
    return 'cms.PSet';
