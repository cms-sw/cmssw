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

process.dqmSaver.workflow = "/CMSSW_3_1_0/RelVal/TrigVal"
process.DQMStore.verbose=0
process.maxEvents.input = 2
process.source.fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_0_0_pre7/RelValTTbar/GEN-SIM-RECO/STARTUP_30X_v1/0002/0AB7BC40-87E9-DD11-8CCF-003048767D51.root'
)


process.load("HLTriggerOffline.Common.HLTValidation_cff")

#extra config needed in standalone
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.validation = cms.Path(
    process.hltvalidation
   # process.HLTMuonVal
   #+process.HLTTauVal
   #+process.EcalPi0Mon
   #+process.EcalPhiSymMon
   #+process.egammaValidationSequence
   #+process.HLTSusyExoVal
   #+process.heavyFlavorValidationSequence
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


