import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process("L1AlgoTest",eras.Phase2_trigger)

process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.MessageLogger.categories = cms.untracked.vstring('L1EGRateStudies', 'FwkReport')
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
   reportEvery = cms.untracked.int32(100)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50) )

process.source = cms.Source("PoolSource",
    # Set to do test run on official Phase-2 L1T Ntuples
    fileNames = cms.untracked.vstring('file:root://cms-xrd-global.cern.ch//store/mc/PhaseIIFall17D/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/L1TPU200_93X_upgrade2023_realistic_v5-v2/30000/564C271B-9654-E811-9338-90B11C2AA16C.root'),
   dropDescendantsOfDroppedBranches=cms.untracked.bool(False),
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
                    "drop l1tHGCalTowerMapBXVector_hgcalTriggerPrimitiveDigiProducer_towerMap_HLT",
                    "drop PCaloHits_g4SimHits_EcalHitsEB_SIM",
                    "drop EBDigiCollection_simEcalUnsuppressedDigis__HLT",
                    "drop PCaloHits_g4SimHits_HGCHitsEE_SIM",
                    "drop HGCalDetIdHGCSampleHGCDataFramesSorted_mix_HGCDigisEE_HLT",

   )
)

# All this stuff just runs the various EG algorithms that we are studying
                         
# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '100X_upgrade2023_realistic_v1', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '93X_upgrade2023_realistic_v5', '')

# Choose a 2030 geometry!
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

# Add HCAL Transcoder
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')





# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters using Emulator

process.load('L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi')



# ----------------------------------------------------------------------------------------------
# 
# ----   Produce the calibrated L1Tower collection

process.L1TowerCalibrationProducer = cms.EDProducer("L1TowerCalibrator",
    # Choosen settings (v8 24 Jan 2019)
    HcalTpEtMin = cms.double(0.0),
    EcalTpEtMin = cms.double(0.0),
    HGCalHadTpEtMin = cms.double(0.25),
    HGCalEmTpEtMin = cms.double(0.25),
    HFTpEtMin = cms.double(0.5),
    puThreshold = cms.double(5.0),
    puThresholdEcal = cms.double(2.0),
    puThresholdHcal = cms.double(3.0),
    puThresholdL1eg = cms.double(4.0),
    puThresholdHGCalEMMin = cms.double(1.0),
    puThresholdHGCalEMMax = cms.double(1.5),
    puThresholdHGCalHadMin = cms.double(0.5),
    puThresholdHGCalHadMax = cms.double(1.0),
    puThresholdHFMin = cms.double(4.0),
    puThresholdHFMax = cms.double(10.0),
    debug = cms.bool(False),
    #debug = cms.bool(True),
    l1CaloTowers = cms.InputTag("L1EGammaClusterEmuProducer","L1CaloTowerCollection","L1AlgoTest"),
    L1HgcalTowersInputTag = cms.InputTag("hgcalTriggerPrimitiveDigiProducer","tower"),
    hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    nHits_to_nvtx_params = cms.VPSet( # Parameters derived on 27 Jan 2019
        cms.PSet(
            fit = cms.string( "hf" ),
            params = cms.vdouble( -0.695, 0.486 )
        ),
        cms.PSet(
            fit = cms.string( "ecal" ),
            params = cms.vdouble( -14.885, 0.666 )
        ),
        cms.PSet(
            fit = cms.string( "hgcalEM" ),
            params = cms.vdouble( -0.334, 0.278 )
        ),
        cms.PSet(
            fit = cms.string( "hgcalHad" ),
            params = cms.vdouble( -1.752, 0.485 )
        ),
        cms.PSet(
            fit = cms.string( "hcal" ),
            params = cms.vdouble( -11.713, 1.574 )
        ),
    )
)

process.pL1Objs = cms.Path( 
    process.L1EGammaClusterEmuProducer *
    process.L1TowerCalibrationProducer
)


process.Out = cms.OutputModule( "PoolOutputModule",
     fileName = cms.untracked.string( "l1TowerCalibrationTest.root" ),
     fastCloning = cms.untracked.bool( False ),
     outputCommands = cms.untracked.vstring(
                          "keep *_L1EGammaClusterEmuProducer_*_*",
                          "keep *_L1TowerCalibrationProducer_*_*",
                          )
)

process.end = cms.EndPath( process.Out )

#dump_file = open("dump_file.py", "w")
#dump_file.write(process.dumpPython())


