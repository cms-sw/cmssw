import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000

process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("HLTriggerOffline.Exotica.HLTExoticaPostVal_cff")
#process.load("HLTriggerOffline.Exotica.HLTExoticaQualityTester_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:hltExoticaValidator.root')
)

process.postprocessor_path = cms.Path(
		process.HLTExoticaPostVal
                #* process.hltExoticaQualityTester
)

process.edmtome_path = cms.Path(process.EDMtoME)
process.dqmsave_path = cms.Path(process.DQMSaver)

process.schedule = cms.Schedule(process.edmtome_path,
                                process.postprocessor_path,
                                process.dqmsave_path)

process.DQMStore.referenceFileName = ''
