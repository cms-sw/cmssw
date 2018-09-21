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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(16) )

process.source = cms.Source("PoolSource",
    # Set to do test run on official Phase-2 L1T Ntuples
    #/GluGluHToTauTau_M125_14TeV_powheg_pythia8/PhaseIIFall17D-L1TnoPU_93X_upgrade2023_realistic_v5-v1/GEN-SIM-DIGI-RAW
    #/store/mc/PhaseIIFall17D/GluGluHToTauTau_M125_14TeV_powheg_pythia8/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/00000/00C160E6-6A39-E811-B904-008CFA152144.root
    #
    #/QCD_Pt-0to1000_Tune4C_14TeV_pythia8/PhaseIIFall17D-L1TnoPU_93X_upgrade2023_realistic_v5-v1/GEN-SIM-DIGI-RAW
    #/store/mc/PhaseIIFall17D/QCD_Pt-0to1000_Tune4C_14TeV_pythia8/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/00000/02AE7A07-2339-E811-B98B-E0071B7AC750.root
    #
    #/WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8/PhaseIIFall17D-L1TnoPU_93X_upgrade2023_realistic_v5-v3/GEN-SIM-DIGI-RAW
    #/store/mc/PhaseIIFall17D/WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v3/30000/162DC63A-C458-E811-92E1-B083FED42FAF.root

    #fileNames = cms.untracked.vstring('file:root://cms-xrd-global.cern.ch//store/mc/PhaseIIFall17D/SingleE_FlatPt-2to100/GEN-SIM-DIGI-RAW/L1TPU200_93X_upgrade2023_realistic_v5-v1/80000/C0F55AFC-1638-E811-9A14-EC0D9A8221EE.root'),
    fileNames = cms.untracked.vstring('file:root://cms-xrd-global.cern.ch//store/mc/PhaseIIFall17D/QCD_Pt-0to1000_Tune4C_14TeV_pythia8/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/00000/02AE7A07-2339-E811-B98B-E0071B7AC750.root'),
    #fileNames = cms.untracked.vstring('file:root://cms-xrd-global.cern.ch//store/mc/PhaseIIFall17D/GluGluHToTauTau_M125_14TeV_powheg_pythia8/GEN-SIM-DIGI-RAW/L1TnoPU_93X_upgrade2023_realistic_v5-v1/00000/00C160E6-6A39-E811-B904-008CFA152144.root'),
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



# --------------------------------------------------------------------------------------------
#
# ----     Produce the L1EGCrystal clusters (code of Sasha Savin)

#process.L1EGammaCrystalsProducer = cms.EDProducer("L1EGCrystalClusterProducer",
#   EtminForStore = cms.double(0.),
#   EcalTpEtMin = cms.untracked.double(0.5), # 500 MeV default per each Ecal TP
#   EtMinForSeedHit = cms.untracked.double(1.0), # 1 GeV decault for seed hit
#   debug = cms.untracked.bool(False),
#   useRecHits = cms.untracked.bool(False),
#   doBremClustering = cms.untracked.bool(True), # Should always be True when using for E/Gamma
#   #ecalTPEB = cms.InputTag("EcalEBTrigPrimProducer","","L1AlgoTest"),
#   ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis","","HLT"),
#   #ecalTPEB = cms.InputTag("EcalTPSorterProducer","EcalTPsTopPerRegion","L1AlgoTest"),
#   ecalRecHitEB = cms.InputTag("ecalRecHit","EcalRecHitsEB","RECO"),
#   #hcalRecHit = cms.InputTag("hbhereco"),
#   hcalTP = cms.InputTag("simHcalTriggerPrimitiveDigis","","HLT"),
#   useTowerMap = cms.untracked.bool(False)
#)

# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters using Emulator

process.load('L1Trigger/L1CaloTrigger/L1EGammaCrystalsEmulatorProducer_cfi')



# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1CaloJets

process.L1CaloTauProducer = cms.EDProducer("L1CaloTauProducer",
    debug = cms.untracked.bool(False),
    l1CaloTowers = cms.InputTag("L1EGammaClusterEmuProducer","L1CaloTowerCollection","L1AlgoTest"),
    #L1CrystalClustersInputTag = cms.InputTag("L1EGammaCrystalsProducer", "L1EGXtalClusterNoCuts", "L1AlgoTest")
    L1CrystalClustersInputTag = cms.InputTag("L1EGammaClusterEmuProducer", "L1EGXtalClusterEmulator", "L1AlgoTest")
)

process.pL1EG = cms.Path( process.L1EGammaClusterEmuProducer * process.L1CaloTauProducer )




process.Out = cms.OutputModule( "PoolOutputModule",
     fileName = cms.untracked.string( "l1egCrystalTest.root" ),
     fastCloning = cms.untracked.bool( False ),
     outputCommands = cms.untracked.vstring(
                          "keep *_L1EGammaClusterEmuProducer_*_*",
                          "keep *_L1CaloTauProducer_*_*",
                          "keep *_TriggerResults_*_*",
                          "keep *_simHcalTriggerPrimitiveDigis_*_*",
                          "keep *_EcalEBTrigPrimProducer_*_*"
                          )
)

process.end = cms.EndPath( process.Out )



dump_file = open("dump_file.py", "w")
dump_file.write(process.dumpPython())


