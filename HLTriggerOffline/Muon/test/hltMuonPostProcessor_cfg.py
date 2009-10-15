import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000

process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("HLTriggerOffline.Muon.hltMuonPostProcessors_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:hltMuonValidator.root')
)

process.hltMuonPostMain.outputFileName = cms.untracked.string(
    'hltMuonPostProcessor.root'
)

process.postprocessorpath = cms.Path(
    process.EDMtoMEConverter *
    process.hltMuonPostProcessors
)

process.DQMStore.referenceFileName = ''
