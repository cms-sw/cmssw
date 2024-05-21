import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("ANALYSIS",Run3)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag

#process.GlobalTag=GlobalTag(process.GlobalTag,'130X_dataRun3_Prompt_v3','')
process.GlobalTag=GlobalTag(process.GlobalTag,'140X_dataRun3_Prompt_v2' ,'')

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HcalIsoTrackX=dict()
    process.MessageLogger.HcalIsoTrack=dict()

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi')
process.towerMakerAll = process.calotowermaker.clone()
process.towerMakerAll.hbheInput = cms.InputTag("hbhereco")
process.towerMakerAll.hoInput = cms.InputTag("none")
process.towerMakerAll.hfInput = cms.InputTag("none")
process.towerMakerAll.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE"))
process.towerMakerAll.AllowMissingInputs = True

process.load('Calibration.HcalCalibAlgos.hcalIsoTrkAnalyzer_cff')
process.hcalIsoTrkAnalyzer.triggers = []
process.hcalIsoTrkAnalyzer.useRaw = 0   # 1 for Raw
process.hcalIsoTrkAnalyzer.ignoreTriggers = True

process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring("/store/data/Run2024B/Commissioning/ALCARECO/HcalCalIsoTrk-PromptReco-v1/000/379/075/00000/2a67bdd1-76ce-4be0-bf72-41584e8f23a7.root"))


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output.root')
)

process.p = cms.Path(process.hcalIsoTrkAnalyzer)

