import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTCOMPARE")

process.load("HLTriggerOffline.Common.HLTValidation_cff")
process.load("HLTriggerOffline.Common.HLTValidationHarvest_cff")
process.load("HLTriggerOffline.Common.HLTValidationQT_cff")

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')  
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("")
)

#process.dqmSaver.convention = 'RelVal'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.dqmSaver.workflow = "/CMSSW_3_1_0/RelVal/TrigVal"
#process.dqmSaver.referenceHandling = cms.untracked.string('skip')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

process.DQMStore.collateHistograms = False
process.DQMStore.verbose=0
process.DQMStore.referenceFileName = "hltReference.root"
#"/build/nuno/test/CMSSW_3_1_X_2009-02-05-0000/src/HltReference.root"


process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE')
)
process.MessageLogger.categories.append('DQMFileSaver')
process.MessageLogger.cout.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )
process.MessageLogger.cerr.DQMFileSaver = cms.untracked.PSet(
       limit = cms.untracked.int32(1000000)
       )

process.source.fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/STARTUP_30X_v1/0001/A42D4BC9-8C16-DE11-8767-003048678B00.root'
)

process.validation = cms.Path(
    process.hltvalidation
    # process.HLTMuonVal
    # process.muonTriggerRateTimeAnalyzer
    #+process.HLTTauVal
    #+process.egammaValidationSequence
    #+process.HLTSusyExoVal
    #+process.heavyFlavorValidationSequence
    #+process.HLTJetMETValSeq
    #+process.HLTAlCaVal

    )

process.post_validation = cms.Path(
    process.hltpostvalidation
    # process.HLTMuonPostVal
    #+process.HLTTauPostVal
    #+process.EgammaPostVal
    #+process.SusyExoPostVal
    #+process.heavyFlavorValidationHarvestingSequence
    #+process.JetMETPostVal
    #+process.HLTAlCaPostVal
    )

process.qt_validation = cms.Path(
    process.hltvalidationqt
    )

process.edmtome = cms.Path(process.EDMtoMEConverter)
process.saver = cms.Path(process.dqmSaver)

process.schedule = cms.Schedule(
    process.validation,
    process.edmtome,
    process.post_validation,
    process.qt_validation,
    process.saver
    )

for filter in (getattr(process,f) for f in process.filters_()):
    if hasattr(filter,"outputFile"):
        filter.outputFile=""
