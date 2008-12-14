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

process.dqmSaver.workflow = "/CMSSW_2_2_0/RelVal/TrigVal"
process.DQMStore.verbose=0
process.maxEvents.input = 2
process.source.fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_2_2_0/RelValZMM/GEN-SIM-RECO/STARTUP_V7_v1/0000/B4BBDB36-21B9-DD11-A047-000423D8F63C.root'
)

process.load("HLTriggerOffline.Common.HLTValidation_cff")
process.validation = cms.Path(
    #process.hltvalidation
     process.HLTTauVal
   #+process.EcalPi0Mon
    +process.egammaValidationSequence
    +process.HLTTopVal
    )

process.post_validation = cms.Path(
    process.hltpostvalidation
    )

process.EDMtoMEconv_and_saver= cms.Path(process.EDMtoMEConverter*process.dqmSaver)

process.schedule = cms.Schedule(
    process.validation,
    process.post_validation,
    process.EDMtoMEconv_and_saver
    )

for filter in (getattr(process,f) for f in process.filters_()):
    if hasattr(filter,"outputFile"):
        filter.outputFile=""


