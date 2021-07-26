import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process("ANALYSIS",Run2_2017)
#process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:run2_data','')
#auto:run3_mc

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

process.load('Calibration.HcalCalibAlgos.HcalIsoTrkAnalyzer_cff')
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
       'file:/eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/MinBias_AlcaReco_2017D_IsotrkFilter.root',
        #'/store/data/Run2018B/JetHT/ALCARECO/HcalCalIsoTrkFilter-PromptReco-v1/000/317/696/00000/D60EC93B-9870-E811-BAF3-FA163E8DA20D.root',
        #'/store/mc/Run3Summer19DR/DoublePion_E-50/GEN-SIM-RECO/2021ScenarioNZSRECONoPU_106X_mcRun3_2021_realistic_v3-v2/270000/22481A14-0F65-E046-809A-C03709C76325.root'                        
        #'root://xrootd.ba.infn.it///store/data/Run2017E/JetHT/ALCARECO/HcalCalIsoTrkFilter-09Aug2019_UL2017-v1/50000/FF19B3B8-39D3-D941-8162-1AA7FB482D48.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output.root')
)

#process.HcalIsoTrkAnalyzer.maxDzPV = 1.0
#process.HcalIsoTrkAnalyzer.minOuterHit =  0
#process.HcalIsoTrkAnalyzer.minLayerCrossed =  0

process.HcalIsoTrkAnalyzer.triggers = []
process.HcalIsoTrkAnalyzer.oldID = [21701, 21603]
process.HcalIsoTrkAnalyzer.newDepth = [2, 4]
process.HcalIsoTrkAnalyzer.hep17 = True
process.HcalIsoTrkAnalyzer.dataType = 0 #0 for jetHT else 1
#process.HcalIsoTrkAnalyzer.maximumEcalEnergy = 100 # set MIP cut  
#process.HcalIsoTrkAnalyzer.useRaw = 2
process.HcalIsoTrkAnalyzer.unCorrect = True

process.HcalIsoTrkAnalyzer.EEHitEnergyThreshold1 = 0.00
process.HcalIsoTrkAnalyzer.EEHitEnergyThreshold2 = 0.00
process.HcalIsoTrkAnalyzer.EEHitEnergyThreshold3 = 0.00
process.HcalIsoTrkAnalyzer.EEHitEnergyThresholdLow = 0.00
# Default
#process.HcalIsoTrkAnalyzer.EBHitEnergyThreshold = 0.08
#process.HcalIsoTrkAnalyzer.EEHitEnergyThreshold0 = 0.30
# Reduced
#process.HcalIsoTrkAnalyzer.EBHitEnergyThreshold = 0.08
#process.HcalIsoTrkAnalyzer.EEHitEnergyThreshold0 = 0.30
# Remove
#process.HcalIsoTrkAnalyzer.EBHitEnergyThreshold = 0.00
#process.HcalIsoTrkAnalyzer.EEHitEnergyThreshold0 = 0.00

process.p = cms.Path(process.HcalIsoTrkAnalyzer)

