import FWCore.ParameterSet.Config as cms
process = cms.Process("B")
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring("file:a.root"))
process.other = cms.EDProducer("OtherThingProducer", thingLabel=cms.untracked.string("thing"))
process.p = cms.Path(process.other)
process.out = cms.OutputModule("PoolOutputModule",
                               fileName=cms.untracked.string("b.root"),
                               outputCommands=cms.untracked.vstring("drop *","keep *_*_*_B"))
process.test = cms.OutputModule("ProvenanceCheckerOutputModule")
process.o = cms.EndPath(process.test+process.out)
