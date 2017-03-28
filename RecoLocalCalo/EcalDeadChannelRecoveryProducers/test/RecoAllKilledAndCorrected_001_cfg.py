import FWCore.ParameterSet.Config as cms

process = cms.Process("DCRec")

process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")        #   Depreciated
process.load("Configuration.Geometry.GeometryIdeal_cff")

process.load("Configuration.EventContent.EventContent_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START53_V10::All')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )


#       *****************************************************************
#                                Input Source                
#       *****************************************************************
process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring(
#       '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-RECO/STARTUP_V8_v1/0000/200EB7E3-90F3-DD11-B1B0-001D09F2432B.root',
#       '/eos/cms/store/relval/CMSSW_5_3_4_cand1/RelValZEE/GEN-SIM-RECO/PU_START53_V10-v1/0003/22521942-41F7-E111-A383-003048D375AA.root',
#      '/store/relval/CMSSW_5_3_4_cand1/RelValZEE/GEN-SIM-RECO/PU_START53_V10-v1/0003/22521942-41F7-E111-A383-003048D375AA.root',
#       'root://eoscms//eos/cms/store/relval/CMSSW_5_3_4_cand1/RelValZEE/GEN-SIM-RECO/PU_START53_V10-v1/0003/22521942-41F7-E111-A383-003048D375AA.root',
        'rfio:/afs/cern.ch/work/i/ikesisog/public/TestFiles/22521942-41F7-E111-A383-003048D375AA.root',
    )
)


#       *****************************************************************
#                                Output Target                
#       *****************************************************************
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Results.root'),
    outputCommands = cms.untracked.vstring('keep *_*_*_DCRec','keep *_*_*_VALIDATION'),
    #SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring("p") ),
)


#       *****************************************************************
#          RecHitsKiller and RecHitRecoveryProducer module for Barrel                  
#       *****************************************************************
process.CreateEBDeadCells = cms.EDProducer("EBChannelKiller", 
   hitProducer          = cms.string("ecalRecHit"),
   hitCollection        = cms.string("EcalRecHitsEB"),
   reducedHitCollection = cms.string("EcalRecHitsEB"),
   KilledHitCollection  = cms.string("KilledEcalRecHitsEB"),
   DeadChannelsFile     = cms.string("EBDeadCellsEach5.txt"),
   KillDeadCells        = cms.bool(True),
)

process.ModCorrectEBDeadCells = cms.EDProducer("EBDeadChannelRecoveryProducers", 
   hitProducer          = cms.string("CreateEBDeadCells"),
   hitCollection        = cms.string("EcalRecHitsEB"),    
   reducedHitCollection = cms.string("EcalRecHitsEB"),
   DeadChannelsFile     = cms.string("EBDeadCellsEach5.txt"),
   Sum8GeVThreshold     = cms.double(8.0),
   CorrectionMethod     = cms.string("NeuralNetworks"),
   CorrectDeadCells     = cms.bool(True),
)


#       *****************************************************************
#          RecHitsKiller and RecHitRecoveryProducer module for EndCap                  
#       *****************************************************************
process.CreateEEDeadCells = cms.EDProducer("EEChannelKiller", 
   hitProducer          = cms.string("ecalRecHit"),
   hitCollection        = cms.string("EcalRecHitsEE"),
   reducedHitCollection = cms.string("EcalRecHitsEE"),
   KilledHitCollection  = cms.string("KilledEcalRecHitsEE"),
   DeadChannelsFile     = cms.string("EEDeadCellsEach5.txt"),
   KillDeadCells        = cms.bool(True),
)

process.ModCorrectEEDeadCells = cms.EDProducer("EEDeadChannelRecoveryProducers", 
   hitProducer          = cms.string("CreateEEDeadCells"),
   hitCollection        = cms.string("EcalRecHitsEE"),	
   reducedHitCollection = cms.string("EcalRecHitsEE"),
   DeadChannelsFile     = cms.string("EEDeadCellsEach5.txt"),
   Sum8GeVThreshold     = cms.double(8.0),
   CorrectionMethod     = cms.string("NeuralNetworks"),
   CorrectDeadCells     = cms.bool(True),
)


#       *****************************************************************
#                                Execution Path                  
#       *****************************************************************
process.p = cms.Path(process.CreateEBDeadCells * process.ModCorrectEBDeadCells + process.CreateEEDeadCells * process.ModCorrectEEDeadCells )
process.outpath = cms.EndPath(process.out)

