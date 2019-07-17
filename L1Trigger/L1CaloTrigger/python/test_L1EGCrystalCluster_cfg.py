import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process("L1AlgoTest",eras.Phase2_trigger)

process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.MessageLogger.categories = cms.untracked.vstring('L1EGRateStudies', 'FwkReport')
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
   reportEvery = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
   # Set to do test run on official Phase-2 L1T Ntuples
   fileNames = cms.untracked.vstring('file:root://cms-xrd-global.cern.ch//store/mc/PhaseIIFall17D/SingleE_FlatPt-2to100/GEN-SIM-DIGI-RAW/L1TPU200_93X_upgrade2023_realistic_v5-v1/80000/C0F55AFC-1638-E811-9A14-EC0D9A8221EE.root'),
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
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '93X_upgrade2023_realistic_v5', '')

# Choose a 2023 geometry!
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

# Add HCAL Transcoder
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')





# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters (code of Sasha Savin)

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

process.pL1EG = cms.Path( process.L1EGammaCrystalsProducer )




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


