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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
   # Set to do test run on official Phase-2 L1T Ntuples
   fileNames = cms.untracked.vstring('/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU200_90X_upgrade2023_realistic_v9-v1/120000/002A4121-132C-E711-87AD-008CFAFBF618.root')
)

# All this stuff just runs the various EG algorithms that we are studying
                         
# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '90X_upgrade2023_realistic_v9', '')

# Choose a 2023 geometry!
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff') # Geom preferred by Phase-2 L1Trig
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')



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




# --------------------------------------------------------------------------------------------
#
# ----    Produce the L1EGCrystal clusters (code of Sasha Savin)


process.EcalTPSorterProducer = cms.EDProducer("EcalTPSorterProducer",
   tpsToKeep = cms.untracked.double(20),
   towerMapName = cms.untracked.string("newMap.json"),
   ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis","","HLT"),
)

process.pL1EG = cms.Path( process.EcalTPSorterProducer )




process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "ecalTpSlimTest.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring(
                    "keep *_TriggerResults_*_*",
                    "keep *_EcalEBTrigPrimProducer_*_*",
                    "keep *_EcalTPSorterProducer_*_*",
                    "keep *_simEcalEBTriggerPrimitiveDigis_*_*"
                    )
)

process.end = cms.EndPath( process.Out )



#dump_file = open("dump_file.py", "w")
#dump_file.write(process.dumpPython())


