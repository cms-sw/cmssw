import FWCore.ParameterSet.Config as cms

process = cms.Process("dqm")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("HLTriggerOffline.Egamma.EgammaValidation_cff")
process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation")                   
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring()
                            )

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.p = cms.EndPath(process.post+process.dqmSaver)

process.testW = cms.Path(process.egammaValidationSequence)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
