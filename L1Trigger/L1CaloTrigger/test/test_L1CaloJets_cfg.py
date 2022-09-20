import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('REPR',eras.Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D41Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D41_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger.categories = cms.untracked.vstring('L1CaloJets', 'FwkReport')
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # dasgoclient --query="dataset dataset=/VBFHToTauTau*/Phase2HLTTDR*/FEVT"
    #fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/mc/PhaseIIMTDTDRAutumn18DR/VBFHToTauTau_M125_14TeV_powheg_pythia8/FEVT/PU200_103X_upgrade2023_realistic_v2-v1/280000/EFC8271A-8026-6A43-AF18-4CB7609B3348.root'),
    fileNames = cms.untracked.vstring('root://cms-xrd-global.cern.ch//store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5/FEVT/PU200_111X_mcRun4_realistic_T15_v1-v1/120000/084C8B72-BC64-DE46-801F-D971D5A34F62.root'),
    inputCommands = cms.untracked.vstring(
                          "keep *",
                          "drop l1tEMTFHitExtras_simEmtfDigis_CSC_HLT",
                          "drop l1tEMTFHitExtras_simEmtfDigis_RPC_HLT",
                          "drop l1tEMTFTrackExtras_simEmtfDigis__HLT",
                          "drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT",
                          "drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT",
                          "drop l1tEMTFHit2016s_simEmtfDigis__HLT",
                          "drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT",
                          "drop l1tEMTFTrack2016s_simEmtfDigis__HLT",
    )
)

# All this stuff just runs the various EG algorithms that we are studying
                                 
# ---- Global Tag :
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '103X_upgrade2023_realistic_v2', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '') 

# Add HCAL Transcoder
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')


process.L1simulation_step = cms.Path(process.SimL1Emulator)



### Based on: L1Trigger/L1TCommon/test/reprocess_test_10_4_0_mtd5.py
### This code is a portion of what is imported and excludes the 'schedule' portion
### of the two lines below.  It makes the test script run!
### from L1Trigger.Configuration.customiseUtils import L1TrackTriggerTracklet
### process = L1TrackTriggerTracklet(process)
process.load('L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff')
process.L1TrackTriggerTracklet_step = cms.Path(process.L1THybridTracksWithAssociators)




# --------------------------------------------------------------------------------------------
#
# ----    Load the L1CaloJet sequence designed to accompany process named "REPR"

process.load('L1Trigger.L1CaloTrigger.L1CaloJets_cff')
process.l1CaloJets = cms.Path(process.L1TCaloJetsSequence)



process.Out = cms.OutputModule( "PoolOutputModule",
     fileName = cms.untracked.string( "l1caloJetTest.root" ),
     fastCloning = cms.untracked.bool( False ),
     outputCommands = cms.untracked.vstring(
                          "keep *_l1tEGammaClusterEmuProducer_*_*",
                          "keep *_l1tCaloJetProducer_*_*",
                          "keep *_l1tCaloJetHTTProducer_*_*",
                          "keep *_TriggerResults_*_*",
                          "keep *_simHcalTriggerPrimitiveDigis_*_*",
                          "keep *_EcalEBTrigPrimProducer_*_*",
                          "keep *_l1tHGCalTowerProducer_*_*"
                          )
)

process.end = cms.EndPath( process.Out )



#dump_file = open("dump_file.py", "w")
#dump_file.write(process.dumpPython())


