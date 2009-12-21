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
        modulesOnPaths.update( (x for x in getattr(process,p).moduleDependencies().iterkeys()))        
    for p in process.endpaths_():
        modulesOnPaths.update( (x for x in getattr(process,p).moduleDependencies().iterkeys()))

    notOnPaths = allMods.difference(modulesOnPaths)
    
    keepModuleNames = set( (x.label_() for x in keepList) )
    
    getRidOf = notOnPaths.difference(keepModuleNames)
    
    for n in getRidOf:
        delattr(process,n)

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
    unittest.main()