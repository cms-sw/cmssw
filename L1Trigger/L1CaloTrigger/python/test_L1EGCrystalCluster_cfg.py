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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # In DAS see: file dataset=/RelValZEE_14/CMSSW_9_0_0_pre2-90X_upgrade2023_realistic_v1_2023D4-v1/GEN-SIM-DIGI-RAW
   #fileNames = cms.untracked.vstring('file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/0ADFD7B5-4277-E611-8E89-0025905A6132.root')
   #fileNames = cms.untracked.vstring('file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_9_0_0_pre2/RelValZEE_14/GEN-SIM-DIGI-RAW/90X_upgrade2023_realistic_v1_2023D4-v1/10000/48F2EE36-04C2-E611-AD16-0CC47A7C360E.root')
   #fileNames = cms.untracked.vstring('file:/data/truggles/step2_ZEE_PU200_10ev_FEVTDEBUGHLT_customHigherPtTrackParticles-RERUN_L1T_TTAssociator_EcalEBtp_HGCtp.root')
   fileNames = cms.untracked.vstring('file:root://eoscms//eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/rekovic/PhaseIIFall16DR82-820_backport_L1TMC_v1.2.2/step2_ZEE_PU200_100ev_FEVTDEBUGHLT_customHigherPtTrackParticles.root')
)

# All this stuff just runs the various EG algorithms that we are studying
                         
# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Choose a 2023 geometry!
process.load('Configuration.Geometry.GeometryExtended2023D7Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff') # Geom used by L1Trig, doesn't have Ecal EndCap, breaks CaloGeomHelper
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')



# --------------------------------------------------------------------------------------------
#
# ----    Produce the ECAL TPs

#process.simEcalEBTriggerPrimitiveDigis = cms.EDProducer("EcalEBTrigPrimProducer",
process.EcalEBTrigPrimProducer = cms.EDProducer("EcalEBTrigPrimProducer",
    BarrelOnly = cms.bool(True),
#    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","ebDigis"),
    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
#    barrelEcalDigis = cms.InputTag("selectDigi","selectedEcalEBDigiCollection"),
    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    Famos = cms.bool(False),
    nOfSamples = cms.int32(1)
)

process.pEcalTPs = cms.Path( process.EcalEBTrigPrimProducer )




# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters (code of Sasha Savin)

# first you need the ECAL RecHIts :
#process.load('Configuration.StandardSequences.Reconstruction_cff')
#process.reconstruction_step = cms.Path( process.calolocalreco )

process.L1EGammaCrystalsProducer = cms.EDProducer("L1EGCrystalClusterProducer",
   EtminForStore = cms.double(0.),
   debug = cms.untracked.bool(False),
   useECalEndcap = cms.bool(False),
   useRecHits = cms.bool(False),
   ecalTPEB = cms.InputTag("EcalEBTrigPrimProducer","","L1AlgoTest"),
   #ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis","","HLT"),
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


