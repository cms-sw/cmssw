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

#process.L1TowerCalibrationProducer = cms.EDProducer("L1TowerCalibrator",
#    # Choosen settings (v8 24 Jan 2019)
#    HcalTpEtMin = cms.double(0.0),
#    EcalTpEtMin = cms.double(0.0),
#    HGCalHadTpEtMin = cms.double(0.25),
#    HGCalEmTpEtMin = cms.double(0.25),
#    HFTpEtMin = cms.double(0.5),
#    puThreshold = cms.double(5.0),
#    puThresholdEcal = cms.double(2.0),
#    puThresholdHcal = cms.double(3.0),
#    puThresholdL1eg = cms.double(4.0),
#    puThresholdHGCalEMMin = cms.double(1.0),
#    puThresholdHGCalEMMax = cms.double(1.5),
#    puThresholdHGCalHadMin = cms.double(0.5),
#    puThresholdHGCalHadMax = cms.double(1.0),
#    puThresholdHFMin = cms.double(4.0),
#    puThresholdHFMax = cms.double(10.0),
#    debug = cms.bool(False),
#    skipCalibrations = cms.bool(False),
#    #debug = cms.bool(True),
#    l1CaloTowers = cms.InputTag("L1EGammaClusterEmuProducer","L1CaloTowerCollection","L1AlgoTest"),
#    L1HgcalTowersInputTag = cms.InputTag("hgcalTriggerPrimitiveDigiProducer","tower"),
#    hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
#    nHits_to_nvtx_params = cms.VPSet( # Parameters derived on 27 Jan 2019
#        cms.PSet(
#            fit = cms.string( "hf" ),
#            params = cms.vdouble( -0.695, 0.486 )
#        ),
#        cms.PSet(
#            fit = cms.string( "ecal" ),
#            params = cms.vdouble( -14.885, 0.666 )
#        ),
#        cms.PSet(
#            fit = cms.string( "hgcalEM" ),
#            params = cms.vdouble( -0.334, 0.278 )
#        ),
#        cms.PSet(
#            fit = cms.string( "hgcalHad" ),
#            params = cms.vdouble( -1.752, 0.485 )
#        ),
#        cms.PSet(
#            fit = cms.string( "hcal" ),
#            params = cms.vdouble( -11.713, 1.574 )
#        ),
#    ),
#	nvtx_to_PU_sub_params = cms.VPSet(
#		cms.PSet(
#			calo = cms.string( "ecal" ),
#			iEta = cms.string( "er1to3" ),
#			params = cms.vdouble( 0.015630, 0.000701 )
#		),
#		cms.PSet(
#			calo = cms.string( "ecal" ),
#			iEta = cms.string( "er4to6" ),
#			params = cms.vdouble( 0.010963, 0.000590 )
#		),
#		cms.PSet(
#			calo = cms.string( "ecal" ),
#			iEta = cms.string( "er7to9" ),
#			params = cms.vdouble( 0.003597, 0.000593 )
#		),
#		cms.PSet(
#			calo = cms.string( "ecal" ),
#			iEta = cms.string( "er10to12" ),
#			params = cms.vdouble( -0.000197, 0.000492 )
#		),
#		cms.PSet(
#			calo = cms.string( "ecal" ),
#			iEta = cms.string( "er13to15" ),
#			params = cms.vdouble( -0.001255, 0.000410 )
#		),
#		cms.PSet(
#			calo = cms.string( "ecal" ),
#			iEta = cms.string( "er16to18" ),
#			params = cms.vdouble( -0.001140, 0.000248 )
#		),
#		cms.PSet(
#			calo = cms.string( "hcal" ),
#			iEta = cms.string( "er1to3" ),
#			params = cms.vdouble( -0.003391, 0.001630 )
#		),
#		cms.PSet(
#			calo = cms.string( "hcal" ),
#			iEta = cms.string( "er4to6" ),
#			params = cms.vdouble( -0.004845, 0.001809 )
#		),
#		cms.PSet(
#			calo = cms.string( "hcal" ),
#			iEta = cms.string( "er7to9" ),
#			params = cms.vdouble( -0.005202, 0.002366 )
#		),
#		cms.PSet(
#			calo = cms.string( "hcal" ),
#			iEta = cms.string( "er10to12" ),
#			params = cms.vdouble( -0.004619, 0.003095 )
#		),
#		cms.PSet(
#			calo = cms.string( "hcal" ),
#			iEta = cms.string( "er13to15" ),
#			params = cms.vdouble( -0.005728, 0.004538 )
#		),
#		cms.PSet(
#			calo = cms.string( "hcal" ),
#			iEta = cms.string( "er16to18" ),
#			params = cms.vdouble( -0.005151, 0.001507 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalEM" ),
#			iEta = cms.string( "er1p4to1p8" ),
#			params = cms.vdouble( -0.020608, 0.004124 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalEM" ),
#			iEta = cms.string( "er1p8to2p1" ),
#			params = cms.vdouble( -0.027428, 0.005488 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalEM" ),
#			iEta = cms.string( "er2p1to2p4" ),
#			params = cms.vdouble( -0.029345, 0.005871 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalEM" ),
#			iEta = cms.string( "er2p4to2p7" ),
#			params = cms.vdouble( -0.028139, 0.005630 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalEM" ),
#			iEta = cms.string( "er2p7to3p1" ),
#			params = cms.vdouble( -0.025012, 0.005005 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalHad" ),
#			iEta = cms.string( "er1p4to1p8" ),
#			params = cms.vdouble( -0.003102, 0.000622 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalHad" ),
#			iEta = cms.string( "er1p8to2p1" ),
#			params = cms.vdouble( -0.003454, 0.000693 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalHad" ),
#			iEta = cms.string( "er2p1to2p4" ),
#			params = cms.vdouble( -0.004145, 0.000831 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalHad" ),
#			iEta = cms.string( "er2p4to2p7" ),
#			params = cms.vdouble( -0.004486, 0.000899 )
#		),
#		cms.PSet(
#			calo = cms.string( "hgcalHad" ),
#			iEta = cms.string( "er2p7to3p1" ),
#			params = cms.vdouble( -0.010332, 0.002068 )
#		),
#		cms.PSet(
#			calo = cms.string( "hf" ),
#			iEta = cms.string( "er29to33" ),
#			params = cms.vdouble( -0.108537, 0.021707 )
#		),
#		cms.PSet(
#			calo = cms.string( "hf" ),
#			iEta = cms.string( "er34to37" ),
#			params = cms.vdouble( -0.102821, 0.020566 )
#		),
#		cms.PSet(
#			calo = cms.string( "hf" ),
#			iEta = cms.string( "er38to41" ),
#			params = cms.vdouble( -0.109859, 0.021974 )
#		)
#	)
#)



# --------------------------------------------------------------------------------------------
#
# ----    Produce the calibrated tower collection combining Barrel, HGCal, HF

process.load('L1Trigger/L1CaloTrigger/L1TowerCalibrationProducer_cfi')



# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1CaloJets

process.load('L1Trigger/L1CaloTrigger/L1CaloJetProducer_cfi')



process.pL1Objs = cms.Path( 
    process.L1EGammaClusterEmuProducer *
    process.L1TowerCalibrationProducer *
    process.L1CaloJetProducer
)


process.Out = cms.OutputModule( "PoolOutputModule",
     fileName = cms.untracked.string( "l1TowerCalibrationTest.root" ),
     fastCloning = cms.untracked.bool( False ),
     outputCommands = cms.untracked.vstring(
                          "keep *_L1EGammaClusterEmuProducer_*_*",
                          "keep *_L1TowerCalibrationProducer_*_*",
                          "keep *_L1CaloJetProducer_*_*",
                          )
)

process.end = cms.EndPath( process.Out )

#dump_file = open("dump_file.py", "w")
#dump_file.write(process.dumpPython())


