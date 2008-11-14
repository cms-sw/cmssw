import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTHARVEST")

process.load("HLTriggerOffline.Common.HLTValidationHarvest_cff")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("")
)

process.DQMStore.collateHistograms = False

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
#End of 'RelVal convention settings

process.dqmSaver.workflow = ""
process.DQMStore.verbose=3

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE')
)

# Other statements

#Adding DQMFileSaver to the message logger configuration
process.MessageLogger.categories.append('DQMFileSaver')
process.MessageLogger.cout.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )
process.MessageLogger.cerr.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )

process.dqmSaver.workflow = "/CMSSW_2_2_0_pre1/RelVal/TrigVal"
process.DQMStore.verbose=0
process.maxEvents.input = -1
process.source.fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_2_2_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP_V7_v1/0000/BC32DA06-1AAF-DD11-9F58-000423D986A8.root',
'/store/relval/CMSSW_2_2_0_pre1/RelValZTT/GEN-SIM-RECO/STARTUP_V7_v1/0000/38F9F4FE-19AF-DD11-BACE-001617E30CD4.root'
)

process.load("HLTriggerOffline.Common.HLTValidation_cff")
process.validation = cms.Path(
    #process.hltvalidation
    process.muonTriggerRateTimeAnalyzer
    )

process.post_validation = cms.Path(
    process.hltpostvalidation
    )

process.EDMtoMEconv_and_saver= cms.Path(process.EDMtoMEConverter*process.dqmSaver)

process.schedule = cms.Schedule(
    #process.validation,
    process.post_validation,
    process.EDMtoMEconv_and_saver
    )

for filter in (getattr(process,f) for f in process.filters_()):
    if hasattr(filter,"outputFile"):
        filter.outputFile=""


