import FWCore.ParameterSet.Config as cms
from copy import deepcopy

# this is a pure Python module to build the deeply nested PSets that describe a 
# SummationSpecification.
# The C++ code assumes the form is fully correct, so you should always use this, 
# which outputs a valid form.

# these need to stay in sync with the C++ enums.  TODO: Can we use Python3 enum or sth.?
NO_TYPE  = cms.int32(0)
GROUPBY  = cms.int32(1)
EXTEND_X = cms.int32(2) 
EXTEND_Y = cms.int32(3)
COUNT    = cms.int32(4)
REDUCE   = cms.int32(5)
SAVE     = cms.int32(6)
CUSTOM   = cms.int32(7)

NO_STAGE = cms.int32(0)
FIRST    = cms.int32(1)
STAGE1   = cms.int32(2)
STAGE1_2 = cms.int32(3)
STAGE2   = cms.int32(4)

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

    if mode == "SUM":
      if self._state == STAGE1:
        if not "Event" in self.spec[0].columns:
          raise Exception("Only per-event counting supported for step1.")
        self.spec[0].columns.remove("Event"); # per-Event groupng is done automatically
        self._state = STAGE1_2
      t = GROUPBY
      cstrings = cms.vstring(cnames)
    elif mode == "EXTEND_X" or mode == "EXTEND_Y":
      if self._state == FIRST: 
        raise Exception("First grouping must be SUM") 
      cname = self._activeColumns.difference(cnames)
      if len(cname) != 1:
        raise Exception("EXTEND must drop exactly one column.")
      cstrings = cms.vstring(cname)

      if mode == "EXTEND_X":
        t = EXTEND_X
      else:
        t = EXTEND_Y
    else:
      raise Exception("Summation mode %s unknown" % mode)

    self._activeColumns = set(cnames)
    self._lastColumns = cnames
    self._lastMode = mode
    
    self.spec.append(cms.PSet(
      type = t, 
      stage = self._state, 
      columns = cstrings,
      arg = cms.string(mode)
    ))

    if self._state == FIRST:
      self._state = STAGE1
    return self

  def save(self):
    if self._state == FIRST:
      raise Exception("First statement must be groupBy.")
    self.spec.append(cms.PSet(
      type = SAVE, 
      stage = self._state, 
      columns = cms.vstring(),
      arg = cms.string("")
    ))
    self._state = STAGE2
    return self

  def custom(self, arg = ""):
    if self.spec[-1].type != SAVE:
      # this is not obvious but needed for now, to avoid memory management
      # trouble with bare TH1's. After save all data is in MEs which are save.
      self.save()
    if self._state != STAGE2:
      raise Exception("Custom processing exists only in Harvesting.")
    self.spec.append(cms.PSet(
      type = CUSTOM, 
      stage = self._state, 
      columns = cms.vstring(),
      arg = cms.string(arg)
    ))
    return self

  def reduce(self, sort):
    if self._state == FIRST:
      raise Exception("First statement must be groupBy.")
    if sort != "MEAN" and sort != "COUNT":
      raise Exception("reduction type %s not known" % sort)
    if self._state == STAGE1:
      if sort == "COUNT":
        t = COUNT
      elif sort == "MEAN":
        t = REDUCE
      else:
        raise Exception("reduction type %s not allowed in step1" % sort)
    else:
      t = REDUCE

    self.spec.append(cms.PSet(
      type = t, 
      stage = self._state, 
      columns = cms.vstring(),
      arg = cms.string(sort)
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
