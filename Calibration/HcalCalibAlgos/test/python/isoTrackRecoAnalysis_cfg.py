import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process("ANALYSIS",Run2_2017)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:run2_data','')

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

process.load('Calibration.HcalCalibAlgos.hcalIsoTrkAnalyzer_cff')
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
       'file:/eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/MinBias_AlcaReco_2017D_IsotrkFilter.root',
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output.root')
)

process.hcalIsoTrkAnalyzer.triggers = []
process.hcalIsoTrkAnalyzer.oldID = [21701, 21603]
process.hcalIsoTrkAnalyzer.newDepth = [2, 4]
process.hcalIsoTrkAnalyzer.hep17 = True
process.hcalIsoTrkAnalyzer.dataType = 0 # 0 for jetHT else 1
process.hcalIsoTrkAnalyzer.maximumEcalEnergy = 2.0 # set MIP cut  
process.hcalIsoTrkAnalyzer.useRaw = 0   # 1 for Raw
process.hcalIsoTrkAnalyzer.unCorrect = True
process.hcalIsoTrkAnalyzer.EEHitEnergyThreshold1 = 0.00 # default 0.30
process.hcalIsoTrkAnalyzer.EEHitEnergyThreshold2 = 0.00 # coeff. of linear term
process.hcalIsoTrkAnalyzer.EEHitEnergyThreshold3 = 0.00 # coeff. of quad term
process.hcalIsoTrkAnalyzer.EEHitEnergyThresholdLow = 0.00 # minimum def 0.30

process.p = cms.Path(process.hcalIsoTrkAnalyzer)

