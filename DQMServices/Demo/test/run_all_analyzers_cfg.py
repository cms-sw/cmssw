import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.DQMStore = cms.Service("DQMStore")
process.MessageLogger = cms.Service("MessageLogger")

process.options = cms.untracked.PSet()
process.options.numberOfThreads = cms.untracked.uint32(1)
process.options.numberOfStreams = cms.untracked.uint32(1)

process.source = cms.Source("EmptySource", numberEventsInRun = cms.untracked.uint32(100),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            firstEvent = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(20))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("DQMServices.Demo.test_cfi")
process.load("DQMServices.Demo.testone_cfi")
process.load("DQMServices.Demo.testonefillrun_cfi")
process.load("DQMServices.Demo.testonelumi_cfi")
process.load("DQMServices.Demo.testonelumifilllumi_cfi")
process.load("DQMServices.Demo.testglobal_cfi")
process.load("DQMServices.Demo.testlegacy_cfi")
process.load("DQMServices.Demo.testlegacyfillrun_cfi")
process.load("DQMServices.Demo.testlegacyfilllumi_cfi")
process.test_reco_dqm = cms.Sequence(process.test 
                                   + process.testone + process.testonefillrun + process.testonelumi + process.testonelumifilllumi 
                                   + process.testglobal 
                                   + process.testlegacy + process.testlegacyfillrun + process.testlegacyfilllumi)

process.p = cms.Path(process.test_reco_dqm)

process.out = cms.OutputModule(
  "DQMRootOutputModule",
  fileName = cms.untracked.string("dqm_out.root"),
  outputCommands = cms.untracked.vstring(
    'keep *'
  )
)

process.o = cms.EndPath(process.out)

