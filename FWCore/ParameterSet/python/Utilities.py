def removeModulesNotOnAPathExcluding( process, keepList=() ):
    """Given a 'process', find all modules (EDProducers,EDFilters,EDAnalyzers,OutputModules)
    and remove them if they do not appear on a Path or EndPath.  One can optionally pass in
    a list of modules which even if they are not on a Path or EndPath you wish to have stay 
    in the configuration [useful for unscheduled execution].
    """
    allMods=set((x for x in process.producers_().iterkeys()))
    allMods.update((x for x in process.filters_().iterkeys()))
    allMods.update((x for x in process.analyzers_().iterkeys()))
    allMods.update((x for x in process.outputModules_().iterkeys()))
    
    modulesOnPaths = set()
    for p in process.paths_():
        modulesOnPaths.update( (x for x in getattr(process,p).moduleNames()))        
    for p in process.endpaths_():
        modulesOnPaths.update( (x for x in getattr(process,p).moduleNames()))

    notOnPaths = allMods.difference(modulesOnPaths)
    
    keepModuleNames = set( (x.label_() for x in keepList) )
    
    getRidOf = notOnPaths.difference(keepModuleNames)
    
    for n in getRidOf:
        delattr(process,n)

class InputTagLabelSet :
  # This holds module labels in a set
  def __init__(self):
    self.labels = set()
  # Get the module labels from InputTags and VInputTags
  def __call__(self, parameter) :
    from FWCore.ParameterSet.Types import VInputTag, InputTag
    if isinstance(parameter, InputTag) :
      self.labels.add(parameter.getModuleLabel())
    elif isinstance(parameter, VInputTag) :
      for (i,inputTagInVInputTag) in enumerate(parameter) :
        if isinstance(inputTagInVInputTag, InputTag) :
          self.labels.add(inputTagInVInputTag.getModuleLabel())
        else :
          # a VInputTag can also hold strings
          self.labels.add(inputTagInVInputTag)

def traverseInputTags(pset, visitor):
  from FWCore.ParameterSet.Mixins import _Parameterizable
  from FWCore.ParameterSet.Types import VPSet, VInputTag, InputTag
  # Loop over parameters in a PSet
  for name in pset.parameters_().keys() :
    value = getattr(pset,name)
    # Recursive calls into a PSet in a PSet
    if isinstance(value, _Parameterizable) :
      traverseInputTags(value, visitor)
    # Recursive calls into PSets in a VPSet
    elif isinstance(value, VPSet) :
      for (n, psetInVPSet) in enumerate(value):
        traverseInputTags(psetInVPSet, visitor)
    # Get the labels from a VInputTag
    elif isinstance(value, VInputTag) :
      visitor(value)
    # Get the label from an InputTag
    elif isinstance(value, InputTag) :
      visitor(value)
    #ignore the rest

def convertToUnscheduled(proc):
  import FWCore.ParameterSet.Config as cms
  """Given a 'Process', convert from scheduled execution to unscheduled. This is done by
    1. Removing all modules not on Paths or EndPaths
    2. Pulling all EDProducers off of all Paths
    3. Dropping any paths which are now empty
    4. Fixing up the Schedule if needed
  """
  proc.prune()
  if not hasattr(proc,'options'):
    proc.options = cms.untracked.PSet()
  proc.options.allowUnscheduled = cms.untracked.bool(True)
  l = proc.paths
  droppedPaths =[]
  #have to get them now since switching them after the
  # paths have been changed gives null labels
  if proc.schedule:
    pathNamesInScheduled = [p.label_() for p in proc.schedule]
  else:
    pathNamesInScheduled = False

  # Look for EDProducers that depend on an EDFilter, either
  # directly or indirectly through another EDProducer. These
  # EDProducers must be in a path and run scheduled. The
  # first loop will find all the direct dependencies, but then
  # the loop must be repeated until no more dependencies
  # are found in order to find the indirect dependencies.
  # Note that here we are assuming that if a module
  # has a configuration parameter that is an InputTag, then
  # it depends on the module with the same module label as in
  # the InputTag. We also assume all dependencies can be found
  # that way. The second assumption is false the modules gets
  # products without using a configurable InputTag. This conversion
  # script does not work properly in those cases, which might result
  # in ProductNotFound exceptions when the converted configuration
  # is run.

  allEDFilters = set(proc.filters_().keys())
  allEDProducers = set(proc.producers_().keys())

  dependentProducers = set()
  firstPass = True

  while True :

    dependentsFoundThisPass = False

    for producer in allEDProducers :
      if producer not in dependentProducers :
        iModule = getattr(proc,producer)

        inputTagLabels = InputTagLabelSet()
        traverseInputTags(getattr(proc,producer), inputTagLabels)

        if firstPass :
          if not inputTagLabels.labels.isdisjoint(allEDFilters) :
            dependentProducers.add(producer)
            dependentsFoundThisPass = True

        if not inputTagLabels.labels.isdisjoint(dependentProducers) :
          dependentProducers.add(producer)
          dependentsFoundThisPass = True

    if not dependentsFoundThisPass :
      break
    firstPass = False

  # Loop over paths
  # On each path we drop EDProducers except we
  # keep the EDProducers that depend on EDFilters
  for pName,p in l.iteritems():
    nodes = []
    v = cms.ModuleNodeVisitor(nodes)
    p.visit(v)
    names = [node.label_() for node in nodes]
    remaining =[]

    for n in names:
      if not isinstance(getattr(proc,n), cms.EDProducer):
        remaining.append(n)
      else :
        if n in dependentProducers :
          remaining.append(n)

    if remaining:
      p=getattr(proc,remaining[0])
      for m in remaining[1:]:
        p+=getattr(proc,m)
      setattr(proc,pName,cms.Path(p))
    else:
      delattr(proc,pName)
      droppedPaths.append(pName)
      
  # If there is a schedule then it needs to be adjusted to drop
  # unneeded paths and also to point at the new Path objects
  # that replaced the old ones.
  if droppedPaths and proc.schedule:
    for p in droppedPaths:
      if p in pathNamesInScheduled:
        pathNamesInScheduled.remove(p)
  if proc.schedule:
    proc.schedule = cms.Schedule([getattr(proc,p) for p in pathNamesInScheduled])


if __name__ == "__main__":
    import unittest
    class TestModuleCommand(unittest.TestCase):
        def setup(self):
            None
        def testConfig(self):
            import FWCore.ParameterSet.Config as cms
            process = cms.Process("Test")

            process.a = cms.EDProducer("A")
            process.b = cms.EDProducer("B")
            process.c = cms.EDProducer("C")

            process.p = cms.Path(process.b*process.c)

            process.d = cms.EDAnalyzer("D")

            process.o = cms.OutputModule("MyOutput")
            process.out = cms.EndPath(process.o)
            removeModulesNotOnAPathExcluding(process,(process.b,))

            self.assert_(not hasattr(process,'a'))
            self.assert_(hasattr(process,'b'))
            self.assert_(hasattr(process,'c'))
            self.assert_(not hasattr(process,'d'))
            self.assert_(hasattr(process,'o'))
        def testNoSchedule(self):
            import FWCore.ParameterSet.Config as cms
            process = cms.Process("TEST")
            process.a = cms.EDProducer("AProd")
            process.b = cms.EDProducer("BProd")
            process.c = cms.EDProducer("CProd")
            process.d = cms.EDProducer("DProd",
                                       par1 = cms.InputTag("f1"))
            process.e = cms.EDProducer("EProd",
                                       par1 = cms.VInputTag(cms.InputTag("x"), cms.InputTag("f1", "y")))
            process.f = cms.EDProducer("FProd",
                                       par1 = cms.VInputTag("x","y","f1"))
            process.g = cms.EDProducer("GProd",
                                       par1 = cms.InputTag("f"))
            process.h = cms.EDProducer("HProd",
                par1 = cms.bool(True),
                par2 = cms.PSet(
                    par21 = cms.VPSet(
                        cms.PSet(foo = cms.bool(True)),
                        cms.PSet(
                            foo4 = cms.PSet(
                                bar = cms.bool(True),
                                foo1 = cms.InputTag("f1"),
                                foo2 = cms.VInputTag(cms.InputTag("x"), cms.InputTag("f11", "y")),
                                foo3 = cms.VInputTag("q","r","s"))
                        )
                    )
                )
            )
            process.f1 = cms.EDFilter("Filter")
            process.f2 = cms.EDFilter("Filter2")
            process.f3 = cms.EDFilter("Filter3")
            process.p1 = cms.Path(process.a+process.b+process.f1+process.d+process.e+process.f+process.g+process.h)
            process.p4 = cms.Path(process.a+process.f2+process.b+process.f1)
            process.p2 = cms.Path(process.a+process.b)
            process.p3 = cms.Path(process.f1)
            convertToUnscheduled(process)
            self.assertEqual(process.options.allowUnscheduled, cms.untracked.bool(True))
            self.assert_(not hasattr(process,'p2'))
            self.assert_(hasattr(process,'a'))
            self.assert_(hasattr(process,'b'))
            self.assert_(not hasattr(process,'c'))
            self.assert_(hasattr(process,'d'))
            self.assert_(hasattr(process,'e'))
            self.assert_(hasattr(process,'f'))
            self.assert_(hasattr(process,'g'))
            self.assert_(hasattr(process,'f1'))
            self.assert_(hasattr(process,'f2'))
            self.assert_(not hasattr(process,'f3'))
            self.assertEqual(process.p1.dumpPython(None),'cms.Path(process.f1+process.d+process.e+process.f+process.g+process.h)\n')
            self.assertEqual(process.p3.dumpPython(None),'cms.Path(process.f1)\n')
            self.assertEqual(process.p4.dumpPython(None),'cms.Path(process.f2+process.f1)\n')
        def testWithSchedule(self):
            import FWCore.ParameterSet.Config as cms
            process = cms.Process("TEST")
            process.a = cms.EDProducer("AProd")
            process.b = cms.EDProducer("BProd")
            process.c = cms.EDProducer("CProd")
            process.d = cms.EDProducer("DProd",
                                       par1 = cms.InputTag("f1"))
            process.e = cms.EDProducer("EProd",
                                       par1 = cms.VInputTag(cms.InputTag("x"), cms.InputTag("f1", "y")))
            process.f = cms.EDProducer("FProd",
                                       par1 = cms.VInputTag("x","y","f1"))
            process.g = cms.EDProducer("GProd",
                                       par1 = cms.InputTag("f"))
            process.f1 = cms.EDFilter("Filter")
            process.f2 = cms.EDFilter("Filter2")
            process.f3 = cms.EDFilter("Filter3")
            process.p1 = cms.Path(process.a+process.b+process.f1+process.d+process.e+process.f+process.g)
            process.p4 = cms.Path(process.a+process.f2+process.b+process.f1)
            process.p2 = cms.Path(process.a+process.b)
            process.p3 = cms.Path(process.f1)
            process.schedule = cms.Schedule(process.p1,process.p4,process.p2,process.p3)
            convertToUnscheduled(process)
            self.assertEqual(process.options.allowUnscheduled,cms.untracked.bool(True))
            self.assert_(not hasattr(process,'p2'))
            self.assert_(hasattr(process,'a'))
            self.assert_(hasattr(process,'b'))
            self.assert_(not hasattr(process,'c'))
            self.assert_(hasattr(process,'d'))
            self.assert_(hasattr(process,'e'))
            self.assert_(hasattr(process,'f'))
            self.assert_(hasattr(process,'g'))
            self.assert_(hasattr(process,'f1'))
            self.assert_(hasattr(process,'f2'))
            self.assert_(not hasattr(process,'f3'))
            self.assertEqual(process.p1.dumpPython(None),'cms.Path(process.f1+process.d+process.e+process.f+process.g)\n')
            self.assertEqual(process.p3.dumpPython(None),'cms.Path(process.f1)\n')
            self.assertEqual(process.p4.dumpPython(None),'cms.Path(process.f2+process.f1)\n')
            self.assertEqual([p for p in process.schedule],[process.p1,process.p4,process.p3])

    unittest.main()
