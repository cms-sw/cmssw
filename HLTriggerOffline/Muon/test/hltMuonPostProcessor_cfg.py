import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000

process.load('Configuration.StandardSequences.EDMtoMEAtJobEnd_cff')
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("HLTriggerOffline.Muon.HLTMuonPostVal_cff")
process.load("HLTriggerOffline.Muon.HLTMuonQualityTester_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:hltMuonValidator.root')
)

process.postprocessor_path = cms.Path(
    process.hltMuonPostProcessors *
    process.hltMuonQualityTester
)

process.edmtome_path = cms.Path(process.EDMtoME)
process.dqmsave_path = cms.Path(process.DQMSaver)

process.schedule = cms.Schedule(process.edmtome_path,
                                process.postprocessor_path,
                                process.dqmsave_path)

process.DQMStore.referenceFileName = ''
