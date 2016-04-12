import FWCore.ParameterSet.Config as cms

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
CUSTOM     = cms.int32(7)

NO_STAGE = cms.int32(0)
FIRST    = cms.int32(1)
STAGE1   = cms.int32(2)
STAGE2   = cms.int32(3)

def val(maybecms):
  if hasattr(maybecms, "value"):
    return maybecms.value()
  else:
    return maybecms

class Specification:
  def __init__(self):
    self.activeColumns = set()
    self.state = FIRST
    self.pset = cms.PSet(spec = cms.VPSet())

  def groupBy(self, cols, mode = "SUM"):
    cnames = val(cols).split("/")

    if mode == "SUM":
      if self.state == STAGE1:
        if not "Event" in self.pset.spec[0].columns:
          raise Exception("Only per-event counting supported for step1.")
        self.pset.spec[0].columns.remove("Event"); # per-Event groupng is done automatically
      t = GROUPBY
      cstrings = cms.vstring(cnames)
    elif mode == "EXTEND_X" or mode == "EXTEND_Y":
      if self.state == FIRST: 
        raise Exception("First grouping must be SUM") 
      cname = self.activeColumns.difference(cnames)
      if len(cname) != 1:
        raise Exception("EXTEND must drop exactly one column.")
      cstrings = cms.vstring(cname)

      if mode == "EXTEND_X":
        t = EXTEND_X
      else:
        t = EXTEND_Y
    else:
      raise Exception("Summation mode %s unknown" % mode)

    self.activeColumns = set(cnames)
    self.lastColumns = cnames
    self.lastMode = mode
    
    self.pset.spec.append(cms.PSet(
      type = t, 
      stage = self.state, 
      columns = cstrings,
      arg = cms.string(mode)
    ))

    if self.state == FIRST:
      self.state = STAGE1
    return self

  def save(self):
    if self.state == FIRST:
      raise Exception("First statement must be groupBy.")
    self.pset.spec.append(cms.PSet(
      type = SAVE, 
      stage = self.state, 
      columns = cms.vstring(),
      arg = cms.string("")
    ))
    self.state = STAGE2
    return self

  def custom(self, arg = ""):
    if self.state != STAGE2:
      raise Exception("Custom processing exists only in Harvesting.")
    self.pset.spec.append(cms.PSet(
      type = CUSTOM, 
      stage = self.state, 
      columns = cms.vstring(),
      arg = cms.string(arg)
    ))
    return self

  def reduce(self, sort):
    if self.state == FIRST:
      raise Exception("First statement must be groupBy.")
    if sort != "MEAN" and sort != "COUNT":
      raise Exception("reduction type %s not known" % sort)
    if self.state == STAGE1:
      if sort != "COUNT":
        raise Exception("reduction type %s not allowed in step1" % sort)
      t = COUNT
    else:
      t = REDUCE

    self.pset.spec.append(cms.PSet(
      type = t, 
      stage = self.state, 
      columns = cms.vstring(),
      arg = cms.string(sort)
    ))
    return self

  def saveAll(self):
    self.save()
    columns = self.lastColumns
    for i in range(len(columns)-1, 0, -1):
      cols = columns[0:i]
      self.groupBy("/".join(cols), self.lastMode)
      self.save()
    return self
    

  def end(self):
    return self.pset


