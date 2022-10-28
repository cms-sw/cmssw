import FWCore.ParameterSet.Config as cms
#from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
#process = cms.Process("RaddamMuon",Run2_2017)

#from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
#process = cms.Process("RaddamMuon",Run2_2018)

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("RaddamMuon",Run3)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")  
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoJets.Configuration.CaloTowersES_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['phase1_2022_realistic']

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Calibration.HcalCalibAlgos.hcalHBHEMuonHighEta_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HBHEMuon=dict()

process.maxEvents = cms.untracked.PSet( 
    input = cms.untracked.int32(100) 
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'file:/eos/cms/store/group/dpg_hcal/comm_hcal/AmanKaur/das_file/FFAE887E-315E-1548-B271-91E0CD7298D6.root'
#        'root://cms-xrd-global.cern.ch//store/mc/Run3Summer19DR/DYToMuMu_M-20_14TeV_pythia8/GEN-SIM-RECO/2021ScenarioNZSRECONoPU_106X_mcRun3_2021_realistic_v3-v2/130000/FFAE887E-315E-1548-B271-91E0CD7298D6.root'
        )
                            )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("muonHighEta.root")
)

process.hcalHBHEMuonHighEta.useRaw = 0
process.hcalHBHEMuonHighEta.unCorrect = True
process.hcalHBHEMuonHighEta.getCharge = True
#process.hcalHBHEMuonHighEta.collapseDepth = False
#process.hcalHBHEMuonHighEta.isItPlan1 = False
#process.hcalHBHEMuonHighEta.ignoreHECorr = False
#process.hcalHBHEMuonHighEta.isItPreRecHit = False
process.hcalHBHEMuonHighEta.maxDepth = 7
process.hcalHBHEMuonHighEta.verbosity = 111
process.hcalTopologyIdeal.MergePosition = False
process.hcalHBHEMuonHighEta.analyzeMuon = True

process.p = cms.Path(process.hcalHBHEMuonHighEta)
