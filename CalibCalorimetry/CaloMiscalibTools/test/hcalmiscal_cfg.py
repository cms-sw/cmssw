import FWCore.ParameterSet.Config as cms

process = cms.Process("hcalminimumbias")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/0291EFD1-DF78-DD11-AE4D-0019DB29C614.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.recalrechit = cms.EDProducer("HcalRecHitRecalib",
    hbheInput = cms.InputTag("hbhereco"),
    RecalibHFHitCollection = cms.string('RecalibHF'),
    hfInput = cms.InputTag("hfreco"),
    hoInput = cms.InputTag("horeco"),
    RecalibHBHEHitCollection = cms.string('RecalibHBHE'),
    Refactor_mean = cms.untracked.double(1.0),
    Refactor = cms.untracked.double(0.5),
    fileNameHcal = cms.untracked.string('hcalmiscalib_0.1.xml'),
    RecalibHOHitCollection = cms.string('RecalibHO')
)

process.MinRecos = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep HBHERecHitsSorted_*_*_*', 
        'keep HORecHitsSorted_*_*_*', 
        'keep HFRecHitsSorted_*_*_*', 
        'keep CaloTowersSorted_*_*_*', 
        'keep *_caloTowers_*_*'),
    fileName = cms.untracked.string('hcalmin_half.root')
)

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.towerMaker.hbheInput = 'recalrechit:RecalibHBHE'
process.towerMaker.hfInput = 'recalrechit:RecalibHF'
process.towerMaker.hoInput = 'recalrechit:RecalibHO'

#process.p = cms.Path(process.recalrechit)
process.p = cms.Path(process.recalrechit*process.caloTowersRec)

process.e = cms.EndPath(process.MinRecos)
