
def ignoreAllFiltersOnPath(path):
  """Given a 'Path', find all EDFilters and wrap them in 'cms.ignore'
  """
  import FWCore.ParameterSet.Config as cms
  from FWCore.ParameterSet.SequenceTypes import _MutatingSequenceVisitor, _UnarySequenceOperator

  class IgnoreFilters(object):
    def __init__(self):
      self.__lastCallUnary = False
      self.onTask = False
    def __call__(self, obj):
      if self.onTask:
        return obj
      elif isinstance(obj,_UnarySequenceOperator):
        self.__lastCallUnary = True
      elif obj.isLeaf() and isinstance(obj, cms.EDFilter) and not self.__lastCallUnary:
        return cms.ignore(obj)
      else:
        self.__lastCallUnary = False
      return obj
  class IgnoreFiltersVisitor(_MutatingSequenceVisitor):
    def __init__(self):
      self.operator = IgnoreFilters()
      self._levelInTasks = 0
      super(type(self),self).__init__(self.operator)
    def enter(self,visitee):
      if isinstance(visitee, cms.Task):
        self._levelInTasks += 1
      self.operator.onTask = (self._levelInTasks > 0)
      super(IgnoreFiltersVisitor,self).enter(visitee)
    def leave(self,visitee):
      if self._levelInTasks > 0:
        if isinstance(visitee, cms.Task):
          self._levelInTasks -= 1
      super(IgnoreFiltersVisitor,self).leave(visitee)

  mutator = IgnoreFiltersVisitor()
  path.visit(mutator)
  if mutator._didApply():
    path._seq = mutator.result(path)[0]
    path._tasks.clear()
    path.associate(*mutator.result(path)[1])
  return path

def convertToUnscheduled(proc):
  print("covertToUnscheduled is deprecated and no longer needed, and will be removed soon. Please update your configuration.")
  return proc


def modulesInSequences(* sequences):
  from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
  modules = []
  for sequence in sequences:
    sequence.visit(ModuleNodeVisitor(modules))
  return modules


def moduleLabelsInSequences(* sequences):
  return [module.label_() for module in modulesInSequences(* sequences)]

def createTaskWithAllProducersAndFilters(process):
  from FWCore.ParameterSet.Config import Task

  l = [ p for p in process.producers.values()]
  l.extend( (f for f in process.filters.values()) )
  return Task(*l)

def convertToSingleModuleEndPaths(process):
    """Remove the EndPaths in the Process with more than one module
    and replace with new EndPaths each with only one module.
    """
    import FWCore.ParameterSet.Config as cms
    toRemove =[]
    added = []
    for n,ep in process.endpaths_().items():
        tsks = []
        ep.visit(cms.TaskVisitor(tsks))

        names = ep.moduleNames()
        if 1 == len(names):
            continue
        toRemove.append(n)
        for m in names:
            epName = m+"_endpath"
            setattr(process,epName,cms.EndPath(getattr(process,m),*tsks))
            added.append(epName)

    s = process.schedule_()
    if s:
        pathNames = [p.label_() for p in s]
        for rName in toRemove:
            pathNames.remove(rName)
        for n in added:
            pathNames.append(n)
        newS = cms.Schedule(*[getattr(process,n) for n in pathNames])
        if s._tasks:
          newS.associate(*s._tasks)
        process.setSchedule_(newS)

    for r in toRemove:
        delattr(process,r)


if __name__ == "__main__":
    import unittest
    class TestModuleCommand(unittest.TestCase):
        def setup(self):
            None
        def testIgnoreFiltersOnPath(self):
            import FWCore.ParameterSet.Config as cms
            process = cms.Process("Test")
            
            process.f1 = cms.EDFilter("F1")
            process.f2 = cms.EDFilter("F2")
            process.f3 = cms.EDFilter("F3")
            process.f4 = cms.EDFilter("F4")
            process.f5 = cms.EDFilter("F5")
            process.f6 = cms.EDFilter("F6")
            process.t1 = cms.Task(process.f5)
            process.t2 = cms.Task(process.f6)
            process.s = cms.Sequence(process.f4, process.t1)
            
            process.p =  cms.Path(process.f1+cms.ignore(process.f2)+process.f3+process.s, process.t2)
            ignoreAllFiltersOnPath(process.p)
            self.assertEqual(process.p.dumpPython(),'cms.Path(cms.ignore(process.f1)+cms.ignore(process.f2)+cms.ignore(process.f3)+cms.ignore(process.f4), process.t1, process.t2)\n')

        def testCreateTaskWithAllProducersAndFilters(self):

            import FWCore.ParameterSet.Config as cms
            process = cms.Process("TEST")

            process.a = cms.EDProducer("AProd")
            process.b = cms.EDProducer("BProd")
            process.c = cms.EDProducer("CProd")

            process.f1 = cms.EDFilter("Filter")
            process.f2 = cms.EDFilter("Filter2")
            process.f3 = cms.EDFilter("Filter3")

            process.out1 = cms.OutputModule("Output1")
            process.out2 = cms.OutputModule("Output2")

            process.analyzer1 = cms.EDAnalyzer("analyzerType1")
            process.analyzer2 = cms.EDAnalyzer("analyzerType2")

            process.task = createTaskWithAllProducersAndFilters(process)
            process.path = cms.Path(process.a, process.task)

            self.assertEqual(process.task.dumpPython(),'cms.Task(process.a, process.b, process.c, process.f1, process.f2, process.f3)\n')
            self.assertEqual(process.path.dumpPython(),'cms.Path(process.a, process.task)\n')

        def testConvertToSingleModuleEndPaths(self):
            import FWCore.ParameterSet.Config as cms
            process = cms.Process("TEST")
            process.a = cms.EDAnalyzer("A")
            process.b = cms.EDAnalyzer("B")
            process.c = cms.EDProducer("C")
            process.ep = cms.EndPath(process.a+process.b,cms.Task(process.c))
            self.assertEqual(process.ep.dumpPython(),'cms.EndPath(process.a+process.b, cms.Task(process.c))\n')
            convertToSingleModuleEndPaths(process)
            self.assertEqual(False,hasattr(process,"ep"))
            self.assertEqual(process.a_endpath.dumpPython(),'cms.EndPath(process.a, cms.Task(process.c))\n')
            self.assertEqual(process.b_endpath.dumpPython(),'cms.EndPath(process.b, cms.Task(process.c))\n')

    unittest.main()
