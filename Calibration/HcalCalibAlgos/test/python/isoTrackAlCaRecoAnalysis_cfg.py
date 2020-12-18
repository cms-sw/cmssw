import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process("ANALYSIS",Run2_2017)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']
#process.GlobalTag.globaltag = 'START53_V15::All'

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
process.HcalIsoTrkAnalyzer.processName  = 'HLTNew1'
process.HcalIsoTrkAnalyzer.producerName = 'ALCAISOTRACK'
process.HcalIsoTrkAnalyzer.moduleName   = 'IsoProd'

process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/work/g/gwalia/calib/dqm/test_29_04/CMSSW_7_5_0_pre2/src/Calibration/HcalCalibAlgos/test/PoolOutput.root'
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output_alca.root')
)

process.p = cms.Path(process.HcalIsoTrkAnalyzer)

