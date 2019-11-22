
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
  import FWCore.ParameterSet.Config as cms
  """Given a 'Process', convert the python configuration from scheduled execution to unscheduled. This is done by
    1. Pulling EDProducers off Path and EndPath Sequences and putting them on an associated Task.
    2. Pulling EDFilters whose filter results are ignored off Path and EndPath Sequences and putting
    them on an associated Task.
    3. Fixing up the Schedule if needed
  """
  # Warning: It is not always possible to convert a configuration
  # where EDProducers are all run on Paths to an unscheduled
  # configuration by modifying only the python configuration.
  # There is more than one kind of pathological condition
  # that can cause this conversion to produce a configuration
  # that gives different results than the original configuration
  # when run under cmsRun. One should view the converted configuration
  # as a thing which needs to be validated. It is possible for there
  # to be pathologies that cannot be resolved by modifying only the
  # python configuration and may require redesign inside the C++ code
  # of the modules and redesign of the logic. For example,
  # an EDAnalyzer might try to get a product and check if the
  # returned handle isValid. Then it could behave differently
  # depending on whether or not the product was present.
  # The behavior when the product is not present could
  # be meaningful and important. In the unscheduled case,
  # the EDProducer will always run and the product could
  # always be there.

  proc.resolve()

  proc=cleanUnscheduled(proc)
  return proc

def cleanUnscheduled(proc):
  import FWCore.ParameterSet.Config as cms

  pathsAndEndPaths = dict(proc.paths)
  pathsAndEndPaths.update( dict(proc.endpaths) )

  #have to get them now since switching them after the
  # paths have been changed gives null labels
  if proc.schedule:
    pathNamesInScheduled = [p.label_() for p in proc.schedule]
  else:
    pathNamesInScheduled = False

  def getUnqualifiedName(name):
    if name[0] in set(['!','-']):
        return name[1:]
    return name

  def getQualifiedModule(name,proc):
    unqual_name = getUnqualifiedName(name)
    p=getattr(proc,unqual_name)
    if unqual_name != name:
      if name[0] == '!':
        p = ~p
      elif name[0] == '-':
        p = cms.ignore(p)
    return p

  # Loop over paths
  # On each path we move EDProducers and EDFilters that
  # are ignored to Tasks
  producerList = list()
  import six
  for pName, originalPath in six.iteritems(pathsAndEndPaths):
    producerList[:] = []
    qualified_names = []
    v = cms.DecoratedNodeNamePlusVisitor(qualified_names)
    originalPath.visit(v)
    remaining =[]

    for n in qualified_names:
      unqual_name = getUnqualifiedName(n)
      mod = getattr(proc,unqual_name)

      #remove EDProducer's and EDFilter's which are set to ignore
      if not (isinstance(mod, cms.EDProducer) or
              (n[0] =='-' and isinstance(mod, cms.EDFilter)) ):
        remaining.append(n)
      else:
        producerList.append(mod)

    taskList = []
    if v.leavesOnTasks():
      taskList.append(cms.Task(*(v.leavesOnTasks())))
    if (producerList):
      taskList.append(cms.Task(*producerList))

    if remaining:
      p = getQualifiedModule(remaining[0],proc)
      for m in remaining[1:]:
        p+=getQualifiedModule(m,proc)
      setattr(proc,pName,type(getattr(proc,pName))(p))
    else:
      setattr(proc,pName,type(getattr(proc,pName))())

    newPath = getattr(proc,pName)
    if (taskList):
      newPath.associate(*taskList)

  # If there is a schedule then it needs to point at
  # the new Path objects
  if proc.schedule:
      listOfTasks = list(proc.schedule._tasks)
      proc.schedule = cms.Schedule([getattr(proc,p) for p in pathNamesInScheduled])
      proc.schedule.associate(*listOfTasks)
  return proc


def modulesInSequences(* sequences):
  from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
  modules = []
  for sequence in sequences:
    sequence.visit(ModuleNodeVisitor(modules))
  return modules


def moduleLabelsInSequences(* sequences):
  return [module.label() for module in modulesInSequences(* sequences)]

def createTaskWithAllProducersAndFilters(process):
  from FWCore.ParameterSet.Config import Task
  import six

  l = [ p for p in six.itervalues(process.producers)]
  l.extend( (f for f in six.itervalues(process.filters)) )
  return Task(*l)

def convertToSingleModuleEndPaths(process):
    """Remove the EndPaths in the Process with more than one module
    and replace with new EndPaths each with only one module.
    """
    import FWCore.ParameterSet.Config as cms
    import six
    toRemove =[]
    added = []
    for n,ep in six.iteritems(process.endpaths_()):
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

        def testNoSchedule(self):
            import FWCore.ParameterSet.Config as cms
            process = cms.Process("TEST")

            process.a = cms.EDProducer("AProd")
            process.b = cms.EDProducer("BProd")
            process.c = cms.EDProducer("CProd")
            process.d = cms.EDProducer("DProd")
            process.m = cms.EDProducer("MProd")
            process.n = cms.EDProducer("NProd")
            process.r = cms.EDProducer("RProd")
            process.s = cms.EDProducer("SProd")

            process.t1 = cms.Task(process.m)
            t2 = cms.Task(process.n)

            process.f1 = cms.EDFilter("Filter")
            process.f2 = cms.EDFilter("Filter2")
            process.f3 = cms.EDFilter("Filter3")
            process.f4 = cms.EDFilter("FIlter4")

            process.out1 = cms.OutputModule("Output1")
            process.out2 = cms.OutputModule("Output2")

            process.analyzer1 = cms.EDAnalyzer("analyzerType1")
            process.analyzer2 = cms.EDAnalyzer("analyzerType2")

            process.p1 = cms.Path(process.a+process.b+process.f1+process.analyzer1+cms.ignore(process.d)+cms.ignore(process.f2))
            process.p4 = cms.Path(process.a+process.f2+process.b+~process.f1+cms.ignore(process.f4))
            process.p2 = cms.Path(process.a+process.b)
            process.p3 = cms.Path(process.f1, process.t1, t2)

            process.t3 = cms.Task(process.r)
            process.t4 = cms.Task(process.s)
            process.s1 = cms.Sequence(~process.a, process.t3)
            process.p5 = cms.Path(process.b + process.s1, process.t4)
            process.end1 = cms.EndPath(process.out1+process.out2+process.analyzer1+process.analyzer2+process.a+process.b+cms.ignore(process.f1))
            process.end2 = cms.EndPath()
            convertToUnscheduled(process)
            self.assert_(hasattr(process,'p2'))
            self.assert_(hasattr(process,'a'))
            self.assert_(hasattr(process,'b'))
            self.assert_(hasattr(process,'c'))
            self.assert_(hasattr(process,'d'))
            self.assert_(hasattr(process,'f1'))
            self.assert_(hasattr(process,'f2'))
            self.assert_(hasattr(process,'f3'))
            self.assert_(hasattr(process,'f4'))
            self.assert_(hasattr(process,'out1'))
            self.assert_(hasattr(process,'out2'))
            self.assert_(hasattr(process,'analyzer1'))
            self.assert_(hasattr(process,'analyzer2'))

            self.assertEqual(process.p1.dumpPython(),'cms.Path(process.f1+process.analyzer1, cms.Task(process.a, process.b, process.d, process.f2))\n')
            self.assertEqual(process.p2.dumpPython(),'cms.Path(cms.Task(process.a, process.b))\n')
            self.assertEqual(process.p3.dumpPython(),'cms.Path(process.f1, cms.Task(process.m, process.n))\n')
            self.assertEqual(process.p4.dumpPython(),'cms.Path(process.f2+~process.f1, cms.Task(process.a, process.b, process.f4))\n')
            self.assertEqual(process.p5.dumpPython(),'cms.Path(cms.Task(process.a, process.b), cms.Task(process.r, process.s))\n')
            self.assertEqual(process.end1.dumpPython(),'cms.EndPath(process.out1+process.out2+process.analyzer1+process.analyzer2, cms.Task(process.a, process.b, process.f1))\n')
            self.assertEqual(process.end2.dumpPython(),'cms.EndPath()\n')

        def testWithSchedule(self):
            import FWCore.ParameterSet.Config as cms
            process = cms.Process("TEST")

            process.a = cms.EDProducer("AProd")
            process.b = cms.EDProducer("BProd")
            process.c = cms.EDProducer("CProd")
            process.d = cms.EDProducer("DProd")
            process.m = cms.EDProducer("MProd")
            process.n = cms.EDProducer("NProd")

            process.t1 = cms.Task(process.m)
            t2 = cms.Task(process.n)

            process.f1 = cms.EDFilter("Filter")
            process.f2 = cms.EDFilter("Filter2")
            process.f3 = cms.EDFilter("Filter3")
            process.f4 = cms.EDFilter("Filter4")

            process.out1 = cms.OutputModule("Output1")
            process.out2 = cms.OutputModule("Output2")

            process.analyzer1 = cms.EDAnalyzer("analyzerType1")
            process.analyzer2 = cms.EDAnalyzer("analyzerType2")

            process.p1 = cms.Path(process.a+process.b+cms.ignore(process.f1)+process.d+process.f2)
            process.p4 = cms.Path(process.a+process.f2+process.b+~process.f1)
            process.p2 = cms.Path(process.a+process.b)
            process.p3 = cms.Path(process.f1)
            process.p5 = cms.Path(process.a+process.f4) #not used on schedule
            process.end1 = cms.EndPath(process.out1+process.out2+process.analyzer1+process.analyzer2+process.a+process.b+cms.ignore(process.f1))
            process.end2 = cms.EndPath()

            process.schedule = cms.Schedule(process.p1,process.p4,process.p2,process.p3,process.end1,process.end2,tasks=[process.t1,t2])
            convertToUnscheduled(process)
            self.assert_(hasattr(process,'p2'))
            self.assert_(hasattr(process,'a'))
            self.assert_(hasattr(process,'b'))
            self.assert_(hasattr(process,'c'))
            self.assert_(hasattr(process,'d'))
            self.assert_(hasattr(process,'f1'))
            self.assert_(hasattr(process,'f2'))
            self.assert_(hasattr(process,'f3'))
            self.assert_(hasattr(process,"f4"))
            self.assert_(hasattr(process,"p5"))

            self.assertEqual(process.p1.dumpPython(),'cms.Path(process.f2, cms.Task(process.a, process.b, process.d, process.f1))\n')
            self.assertEqual(process.p2.dumpPython(),'cms.Path(cms.Task(process.a, process.b))\n')
            self.assertEqual(process.p3.dumpPython(),'cms.Path(process.f1)\n')
            self.assertEqual(process.p4.dumpPython(),'cms.Path(process.f2+~process.f1, cms.Task(process.a, process.b))\n')
            self.assertEqual(process.p5.dumpPython(),'cms.Path(process.f4, cms.Task(process.a))\n')
            self.assertEqual(process.end1.dumpPython(),'cms.EndPath(process.out1+process.out2+process.analyzer1+process.analyzer2, cms.Task(process.a, process.b, process.f1))\n')
            self.assertEqual(process.end2.dumpPython(),'cms.EndPath()\n')

            self.assertEqual([p for p in process.schedule],[process.p1,process.p4,process.p2,process.p3,process.end1,process.end2])
            listOfTasks = list(process.schedule._tasks)
            self.assertEqual(listOfTasks, [process.t1,t2])

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
