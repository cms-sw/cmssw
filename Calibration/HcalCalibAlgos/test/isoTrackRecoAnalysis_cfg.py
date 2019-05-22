import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process("ANALYSIS",Run2_2017)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag='101X_dataRun2_Prompt_v10'

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('HcalIsoTrack')

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi')
process.towerMakerAll = process.calotowermaker.clone()
process.towerMakerAll.hbheInput = cms.InputTag("hbhereco")
process.towerMakerAll.hoInput = cms.InputTag("none")
process.towerMakerAll.hfInput = cms.InputTag("none")
process.towerMakerAll.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE"))
process.towerMakerAll.AllowMissingInputs = True

process.load('Calibration.HcalCalibAlgos.HcalIsoTrkAnalyzer_cff')
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
        '/store/data/Run2018B/JetHT/ALCARECO/HcalCalIsoTrkFilter-PromptReco-v1/000/317/696/00000/D60EC93B-9870-E811-BAF3-FA163E8DA20D.root',
#       'file:/afs/cern.ch/work/s/sdey/public/forsunandada/C2F61205-A366-E711-9AFA-02163E01A2B0.root',
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output.root')
)

process.HcalIsoTrkAnalyzer.triggers = []
process.HcalIsoTrkAnalyzer.dataType = 0 #0 for jetHT else 1
#process.HcalIsoTrkAnalyzer.useRaw = 2

process.p = cms.Path(process.HcalIsoTrkAnalyzer)

