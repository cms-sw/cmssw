import FWCore.ParameterSet.Config as cms

process = cms.Process("DCRec")

process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")        #   Depreciated
process.load("Configuration.Geometry.GeometryIdeal_cff")

process.load("Configuration.EventContent.EventContent_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('START53_V10::All')
process.GlobalTag.globaltag = 'GR_P_V42_AN2::All'  # this one for run2012D

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )


#       *****************************************************************
#                                Input Source                
#       *****************************************************************
process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
#       '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-RECO/STARTUP_V8_v1/0000/200EB7E3-90F3-DD11-B1B0-001D09F2432B.root',
#       '/eos/cms/store/relval/CMSSW_5_3_4_cand1/RelValZEE/GEN-SIM-RECO/PU_START53_V10-v1/0003/22521942-41F7-E111-A383-003048D375AA.root',
#       '/store/relval/CMSSW_5_3_4_cand1/RelValZEE/GEN-SIM-RECO/PU_START53_V10-v1/0003/22521942-41F7-E111-A383-003048D375AA.root',
#       'file:/afs/cern.ch/work/v/vgiakoum/public/8200EF9B-0AA0-E111-9E58-003048FFCB6A.root'
#    'file:/afs/cern.ch/work/i/ikesisog/public/TestFiles/8200EF9B-0AA0-E111-9E58-003048FFCB6A.root'
    '/store/data/Run2012D/DoublePhotonHighPt/AOD/PromptReco-v1/000/203/994/30ABB9D1-790E-E211-AFF1-001D09F242EF.root',
#    '/store/data/Run2012D/DoublePhotonHighPt/AOD/PromptReco-v1/000/203/994/3E8BBC89-620E-E211-9185-001D09F25479.root',
#    '/store/data/Run2012D/DoublePhotonHighPt/AOD/PromptReco-v1/000/203/994/DC93F9B0-6B0E-E211-8F48-003048D37560.root',
#    '/store/data/Run2012D/DoublePhotonHighPt/AOD/PromptReco-v1/000/203/994/FA29ECAC-640E-E211-A95F-001D09F28D54.root',
#       'root://eoscms//eos/cms/store/relval/CMSSW_5_3_4_cand1/RelValZEE/GEN-SIM-RECO/PU_START53_V10-v1/0003/22521942-41F7-E111-A383-003048D375AA.root',
#    'rfio:/afs/cern.ch/user/i/ikesisog/public/22521942-41F7-E111-A383-003048D375AA.root',
    )
)


#       *****************************************************************
#                                Output Target                
#       *****************************************************************
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Results_vic.root'),
    outputCommands = cms.untracked.vstring('keep EcalRecHitsSorted_*_*_*'),
    #SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring("p") ),
)


#       *****************************************************************
#          RecHitsKiller and RecHitRecoveryProducer module for Barrel                  
#       *****************************************************************
process.CreateEBDeadCells = cms.EDProducer("EBChannelKiller", 
   hitTag               = cms.InputTag("reducedEcalRecHitsEB", ""),
   reducedHitCollection = cms.string("CreateEB"),
   KilledHitCollection  = cms.string("KilledEcalRecHitsEB"),
   DeadChannelsFile     = cms.string("EBDeadCellsEach5.txt"),
   KillDeadCells        = cms.bool(True),
)

process.ModCorrectEBDeadCells = cms.EDProducer("EBDeadChannelRecoveryProducers", 
   hitTag               = cms.InputTag("CreateEBDeadCells", "CreateEB"),
   reducedHitCollection = cms.string("ModifyEB"),
   DeadChannelsFile     = cms.string("EBDeadCellsEach5.txt"),
   Sum8GeVThreshold     = cms.double(8.0),
   CorrectionMethod     = cms.string("NeuralNetworks"),
   CorrectDeadCells     = cms.bool(True),
)


#       *****************************************************************
#          RecHitsKiller and RecHitRecoveryProducer module for EndCap                  
#       *****************************************************************
process.CreateEEDeadCells = cms.EDProducer("EEChannelKiller", 
   hitTag               = cms.InputTag("reducedEcalRecHitsEE", ""),
   reducedHitCollection = cms.string("CreateEE"),
   KilledHitCollection  = cms.string("KilledEcalRecHitsEE"),
   DeadChannelsFile     = cms.string("EEDeadCellsEach5.txt"),
   KillDeadCells        = cms.bool(True),
)

process.ModCorrectEEDeadCells = cms.EDProducer("EEDeadChannelRecoveryProducers", 
   hitTag               = cms.InputTag("CreateEEDeadCells", "CreateEE"),
   reducedHitCollection = cms.string("ModifyEE"),
   DeadChannelsFile     = cms.string("EEDeadCellsEach5.txt"),
   Sum8GeVThreshold     = cms.double(8.0),
   CorrectionMethod     = cms.string("NeuralNetworks"),
   CorrectDeadCells     = cms.bool(True),
)

process.TFileService = cms.Service("TFileService", fileName = cms.string('recovery_hist.root'))

process.validateRecoveryEB = cms.EDAnalyzer("EcalDeadChannelRecoveryAnalyzer",
  originalRecHitCollection = cms.InputTag("reducedEcalRecHitsEB", ""),
  recoveredRecHitCollection = cms.InputTag("ModCorrectEBDeadCells", "ModifyEB"),

  titlePrefix = cms.string("(EB) "),
)

process.validateRecoveryEE = cms.EDAnalyzer("EcalDeadChannelRecoveryAnalyzer",
  originalRecHitCollection = cms.InputTag("reducedEcalRecHitsEE", ""),
  recoveredRecHitCollection = cms.InputTag("ModCorrectEEDeadCells", "ModifyEE"),

  titlePrefix = cms.string("(EE) "),
)

process.dump = cms.EDAnalyzer("EcalRecHitDump",
  EBRecHitCollection = cms.InputTag("ModCorrectEBDeadCells", "ModifyEB"),
  EERecHitCollection = cms.InputTag("ModCorrectEEDeadCells", "ModifyEE"),
)



#       *****************************************************************
#                                Execution Path                  
#       *****************************************************************
process.p = cms.Path(process.CreateEBDeadCells * process.ModCorrectEBDeadCells * process.validateRecoveryEB +
    process.CreateEEDeadCells * process.ModCorrectEEDeadCells * process.validateRecoveryEE ) 

process.outpath = cms.EndPath(process.out)

