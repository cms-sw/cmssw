import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("ANALYSIS",Run2_2018)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

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
process.hcalIsoTrkAnalyzer.triggers = []
process.hcalIsoTrkAnalyzer.useRaw = 0   # 1 for Raw
process.hcalIsoTrkAnalyzer.ignoreTriggers = True

process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
                                'file:/eos/uscms/store/user/lpcrutgers/huiwang/HCAL/UL_DoublePion_E-50_RECO_DLPHIN_zeroOut_noPU-2021-09-21/MC_RECO_0.root',
                                'file:/eos/uscms/store/user/lpcrutgers/huiwang/HCAL/UL_DoublePion_E-50_RECO_DLPHIN_zeroOut_noPU-2021-09-21/MC_RECO_1.root'
                            )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output_alca.root')
)

process.p = cms.Path(process.hcalIsoTrkAnalyzer)

