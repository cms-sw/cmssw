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
  
  for pName,p in l.iteritems():
    nodes = []
    v = cms.ModuleNodeVisitor(nodes)
    p.visit(v)
    names = [node.label_() for node in nodes]
    remaining =[]
    for n in names:
      if not isinstance(getattr(proc,n), cms.EDProducer):
        remaining.append(n)
    if remaining:
      p=getattr(proc,remaining[0])
      for m in remaining[1:]:
        p+=getattr(proc,m)
      setattr(proc,pName,cms.Path(p))
    else:
      delattr(proc,pName)
      droppedPaths.append(pName)
  if droppedPaths and proc.schedule:
    for p in droppedPaths:
      if p in pathNamesInScheduled:
        pathNamesInScheduled.remove(p)
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
            process.f1 = cms.EDFilter("Filter")
            process.f2 = cms.EDFilter("Filter2")
            process.p1 = cms.Path(process.a+process.b+process.f1)
            process.p4 = cms.Path(process.a+process.f2+process.b+process.f1)
            process.p2 = cms.Path(process.a+process.b)
            process.p3 = cms.Path(process.f1)
            convertToUnscheduled(process)
            self.assertEqual(process.options.allowUnscheduled, cms.untracked.bool(True))
            self.assert_(not hasattr(process,'p2'))
            self.assert_(hasattr(process,'a'))
            self.assert_(hasattr(process,'b'))
            self.assert_(hasattr(process,'f1'))
            self.assert_(hasattr(process,'f2'))
            self.assertEqual(process.p1.dumpPython(None),'cms.Path(process.f1)\n')
            self.assertEqual(process.p3.dumpPython(None),'cms.Path(process.f1)\n')
            self.assertEqual(process.p4.dumpPython(None),'cms.Path(process.f2+process.f1)\n')
        def testWithSchedule(self):
            import FWCore.ParameterSet.Config as cms
            process = cms.Process("TEST")
            process.a = cms.EDProducer("AProd")
            process.b = cms.EDProducer("BProd")
            process.f1 = cms.EDFilter("Filter")
            process.f2 = cms.EDFilter("Filter2")
            process.p1 = cms.Path(process.a+process.b+process.f1)
            process.p4 = cms.Path(process.a+process.f2+process.b+process.f1)
            process.p2 = cms.Path(process.a+process.b)
            process.p3 = cms.Path(process.f1)
            process.schedule = cms.Schedule(process.p1,process.p4,process.p2,process.p3)
            convertToUnscheduled(process)
            self.assertEqual(process.options.allowUnscheduled,cms.untracked.bool(True))
            self.assert_(not hasattr(process,'p2'))
            self.assert_(hasattr(process,'a'))
            self.assert_(hasattr(process,'b'))
            self.assert_(hasattr(process,'f1'))
            self.assert_(hasattr(process,'f2'))
            self.assertEqual(process.p1.dumpPython(None),'cms.Path(process.f1)\n')
            self.assertEqual(process.p3.dumpPython(None),'cms.Path(process.f1)\n')
            self.assertEqual(process.p4.dumpPython(None),'cms.Path(process.f2+process.f1)\n')
            self.assertEqual([p for p in process.schedule],[process.p1,process.p4,process.p3])

    unittest.main()