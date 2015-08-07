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

def traverseInputTags(pset, visitor, stringInputLabels):
  from FWCore.ParameterSet.Mixins import _Parameterizable
  from FWCore.ParameterSet.Types import VPSet, VInputTag, InputTag, string

  # Loop over parameters in a PSet
  for name in pset.parameters_().keys() :
    value = getattr(pset,name)
    # Recursive calls into a PSet in a PSet
    if isinstance(value, _Parameterizable) :
      traverseInputTags(value, visitor, stringInputLabels)
    # Recursive calls into PSets in a VPSet
    elif isinstance(value, VPSet) :
      for (n, psetInVPSet) in enumerate(value):
        traverseInputTags(psetInVPSet, visitor, stringInputLabels)
    # Get the labels from a VInputTag
    elif isinstance(value, VInputTag) :
      visitor(value)
    # Get the label from an InputTag
    elif isinstance(value, InputTag) :
      visitor(value)
    # Known module labels in string objects
    elif stringInputLabels and isinstance(value, string) and name in stringInputLabels and value.value() :
      visitor.labels.add(value.value())
    #ignore the rest

def convertToUnscheduled(proc):
  import FWCore.ParameterSet.Config as cms
  """Given a 'Process', convert the python configuration from scheduled execution to unscheduled. This is done by
    1. Removing all modules not on Paths or EndPaths
    2. Pulling EDProducers not dependent on EDFilters off of all Paths
    3. Dropping any Paths which are now empty
    4. Fixing up the Schedule if needed
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

  # Remove all modules not on Paths or EndPaths
  proc.prune()

  # Turn on unschedule mode
  if not hasattr(proc,'options'):
    proc.options = cms.untracked.PSet()
  proc.options.allowUnscheduled = cms.untracked.bool(True)

  proc=cleanUnscheduled(proc)
  return proc

def cleanUnscheduled(proc):
  import FWCore.ParameterSet.Config as cms
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
  # EDProducers must stay in a path and run scheduled. The
  # first loop will find all the direct dependencies, but then
  # the loop must be repeated until no more dependencies
  # are found in order to find the indirect dependencies.
  # Note that here we are assuming that if a module
  # has a configuration parameter that is an InputTag, then
  # it depends on the module with the same module label as in
  # the InputTag. In addition, there are a number of special
  # cases where we have identified specific types of modules
  # which use particular string parameters like InputTag
  # module labels. And so we have some special code to
  # handle those cases also. If there are other cases where
  # the module gets things without using InputTags, then this
  # conversion script can fail, which might result in ProductNotFound
  # exceptions or other problems when the converted configuration is run.

  # The dictionary keys are are the types of the EDProducers
  # The dictionary values are lists of parameter names that are strings
  # used like InputTags.
  knownStringInputLabels = {}
  knownStringInputLabels['KProd'] = ['xSrc'] # a fake entry for a unit test below
  knownStringInputLabels['SeedGeneratorFromRegionHitsEDProducer'] = ['vertexSrc']
  knownStringInputLabels['PixelTrackProducer'] = ['vertexSrc']
  knownStringInputLabels['PixelTracksProducer'] = ['vertexSrc']
  knownStringInputLabels['SimpleTrackListMerger'] = ['TrackProducer1', 'TrackProducer2']

  allEDFilters = set(proc.filters_().keys())
  allEDProducers = set(proc.producers_().keys())

  dependentProducers = set()
  firstPass = True


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

  while True :

    dependentsFoundThisPass = False

    for producer in allEDProducers :
      if producer not in dependentProducers :
        iModule = getattr(proc,producer)

        stringInputLabels = []
        moduleType = iModule.type_()
        if moduleType in knownStringInputLabels :
          stringInputLabels = knownStringInputLabels[moduleType]

        inputTagLabels = InputTagLabelSet()
        traverseInputTags(getattr(proc,producer), inputTagLabels, stringInputLabels)

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
    qualified_names = []
    v = cms.DecoratedNodeNameVisitor(qualified_names)
    p.visit(v)
    remaining =[]

    for n in qualified_names:
      unqual_name = getUnqualifiedName(n)
      if not isinstance(getattr(proc,unqual_name), cms.EDProducer):
        remaining.append(n)
      else :
        if unqual_name in dependentProducers :
          remaining.append(n)

    if remaining:
      p = getQualifiedModule(remaining[0],proc)
      for m in remaining[1:]:
        p+=getQualifiedModule(m,proc)
      setattr(proc,pName,cms.Path(p))
    # drop empty paths
    else:
      setattr(proc,pName,cms.Path())

  # If there is a schedule then it needs to point at
  # the new Path objects
  if proc.schedule:
      proc.schedule = cms.Schedule([getattr(proc,p) for p in pathNamesInScheduled])
  return proc

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
            process.k = cms.EDProducer("KProd",
                par1 = cms.bool(True),
                par2 = cms.PSet(
                    par21 = cms.VPSet(
                        cms.PSet(foo = cms.bool(True)),
                        cms.PSet(
                            foo4 = cms.PSet(
                                bar = cms.bool(True),
                                xSrc = cms.string("f")
                           )
                        )
                    )
                )
            )
            process.f1 = cms.EDFilter("Filter")
            process.f2 = cms.EDFilter("Filter2")
            process.f3 = cms.EDFilter("Filter3")
            process.f4 = cms.EDFilter("FIlter4")
            process.p1 = cms.Path(process.a+process.b+process.f1+process.d+process.e+process.f+process.g+process.h+process.k)
            process.p4 = cms.Path(process.a+process.f2+process.b+~process.f1+cms.ignore(process.f4))
            process.p2 = cms.Path(process.a+process.b)
            process.p3 = cms.Path(process.f1)
            convertToUnscheduled(process)
            self.assertEqual(process.options.allowUnscheduled, cms.untracked.bool(True))
            self.assert_(hasattr(process,'p2'))
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
            self.assert_(hasattr(process,'f4'))
            self.assertEqual(process.p1.dumpPython(None),'cms.Path(process.f1+process.d+process.e+process.f+process.g+process.h+process.k)\n')
            self.assertEqual(process.p2.dumpPython(None),'cms.Path()\n')
            self.assertEqual(process.p3.dumpPython(None),'cms.Path(process.f1)\n')
            self.assertEqual(process.p4.dumpPython(None),'cms.Path(process.f2+~process.f1+cms.ignore(process.f4))\n')
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
            self.assert_(hasattr(process,'p2'))
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
            self.assertEqual(process.p2.dumpPython(None),'cms.Path()\n')
            self.assertEqual(process.p3.dumpPython(None),'cms.Path(process.f1)\n')
            self.assertEqual(process.p4.dumpPython(None),'cms.Path(process.f2+process.f1)\n')
# there is no longer a schedule. 
#            self.assertEqual([p for p in process.schedule],[process.p1,process.p4,process.p2,process.p3])

    unittest.main()
