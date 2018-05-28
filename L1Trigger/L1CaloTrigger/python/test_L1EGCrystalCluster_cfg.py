import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process("L1AlgoTest",eras.phase2_common)

process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.MessageLogger.categories = cms.untracked.vstring('L1EGRateStudies', 'FwkReport')
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
   reportEvery = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",
   # Set to do test run on official Phase-2 L1T Ntuples
   fileNames = cms.untracked.vstring('file:root://cmsxrootd.fnal.gov///store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU200_90X_upgrade2023_realistic_v9-v1/120000/04B4BF1D-1E2C-E711-BE1C-7845C4FC39D1.root'),
   inputCommands = cms.untracked.vstring(
                    "keep *",
                    "drop l1tEMTFHitExtras_simEmtfDigis_CSC_HLT",
                    "drop l1tEMTFHitExtras_simEmtfDigis_RPC_HLT",
                    "drop l1tEMTFTrackExtras_simEmtfDigis__HLT",
   )
)

# All this stuff just runs the various EG algorithms that we are studying
                         
# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '100X_upgrade2023_realistic_v1', '')

# Choose a 2023 geometry!
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')



# --------------------------------------------------------------------------------------------
#
# ----    Produce the ECAL TPs
#
##process.simEcalEBTriggerPrimitiveDigis = cms.EDProducer("EcalEBTrigPrimProducer",
#process.EcalEBTrigPrimProducer = cms.EDProducer("EcalEBTrigPrimProducer",
#    BarrelOnly = cms.bool(True),
##    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","ebDigis"),
#    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
##    barrelEcalDigis = cms.InputTag("selectDigi","selectedEcalEBDigiCollection"),
#    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
#    TcpOutput = cms.bool(False),
#    Debug = cms.bool(False),
#    Famos = cms.bool(False),
#    nOfSamples = cms.int32(1)
#)
#
#process.pEcalTPs = cms.Path( process.EcalEBTrigPrimProducer )

process.EcalTPSorterProducer = cms.EDProducer("EcalTPSorterProducer",
   tpsToKeep = cms.untracked.double(20),
   towerMapName = cms.untracked.string("newMap.json"),
   ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis","","HLT"),
)


# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters (code of Sasha Savin)

# first you need the ECAL RecHIts :
#process.load('Configuration.StandardSequences.Reconstruction_cff')
#process.reconstruction_step = cms.Path( process.calolocalreco )

process.L1EGammaCrystalsProducer = cms.EDProducer("L1EGCrystalClusterProducer",
   EtminForStore = cms.double(0.),
   EcalTpEtMin = cms.untracked.double(0.5), # 500 MeV default per each Ecal TP
   EtMinForSeedHit = cms.untracked.double(1.0), # 1 GeV decault for seed hit
   debug = cms.untracked.bool(False),
   useRecHits = cms.untracked.bool(False),
   doBremClustering = cms.untracked.bool(True), # Should always be True when using for E/Gamma
   #ecalTPEB = cms.InputTag("EcalEBTrigPrimProducer","","L1AlgoTest"),
   ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis","","HLT"),
   #ecalTPEB = cms.InputTag("EcalTPSorterProducer","EcalTPsTopPerRegion","L1AlgoTest"),
   ecalRecHitEB = cms.InputTag("ecalRecHit","EcalRecHitsEB","RECO"),
   hcalRecHit = cms.InputTag("hbhereco"),
   hcalTP = cms.InputTag("simHcalTriggerPrimitiveDigis","","HLT"),
   useTowerMap = cms.untracked.bool(False)
)

process.pL1EG = cms.Path( process.EcalTPSorterProducer + process.L1EGammaCrystalsProducer )




process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "l1egCrystalTest.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring(
                    "keep *_L1EGammaCrystalsProducer_*_*",
                    "keep *_TriggerResults_*_*",
                    "keep *_simHcalTriggerPrimitiveDigis_*_*",
                    "keep *_EcalEBTrigPrimProducer_*_*"
                    )
)

process.end = cms.EndPath( process.Out )



dump_file = open("dump_file.py", "w")
dump_file.write(process.dumpPython())


