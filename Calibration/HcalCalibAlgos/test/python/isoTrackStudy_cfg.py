import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
#process = cms.Process("ANALYSIS",Run2_2017)

#from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
#process = cms.Process("ANALYSIS",Run2_2018)

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("ANALYSIS",Run3)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag='106X_mcRun3_2021_realistic_v3'

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HcalIsoTrack=dict()

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi')
process.towerMakerAll = process.calotowermaker.clone()
process.towerMakerAll.hbheInput = cms.InputTag("hbhereco")
process.towerMakerAll.hoInput = cms.InputTag("none")
process.towerMakerAll.hfInput = cms.InputTag("none")
process.towerMakerAll.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE"))
process.towerMakerAll.AllowMissingInputs = True

process.load('Calibration.HcalCalibAlgos.hcalIsoTrackStudy_cff')
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
        #'/store/data/Run2018B/JetHT/ALCARECO/HcalCalIsoTrkFilter-PromptReco-v1/000/317/696/00000/D60EC93B-9870-E811-BAF3-FA163E8DA20D.root',
        #'/store/mc/Run3Summer19DR/DoublePion_E-50/GEN-SIM-RECO/2021ScenarioNZSRECONoPU_106X_mcRun3_2021_realistic_v3-v2/270000/22481A14-0F65-E046-809A-C03709C76325.root'
        'file:/afs/cern.ch/work/d/dbhowmik/public/etaPlus_input.root',
#       'file:/eos/user/d/dbhowmik/Results_2019/etaPlus_input.root',
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output.root')
)

process.hcalIsoTrackStudy.maxDzPV = 1.0
process.hcalIsoTrackStudy.minOuterHit =  0
process.hcalIsoTrackStudy.minLayerCrossed =  0
process.hcalIsoTrackStudy.triggers = []
process.hcalIsoTrackStudy.dataType = 1 #0 for jetHT else 1
process.hcalIsoTrackStudy.maximumEcalEnergy = 100 # set MIP cut  
#process.hcalIsoTrackStudy.useRaw = 2

process.p = cms.Path(process.hcalIsoTrackStudy)

