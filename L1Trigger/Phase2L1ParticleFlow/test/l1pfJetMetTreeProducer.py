import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("IN", eras.phase2_trigger)
process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '100X_upgrade2023_realistic_v1', '')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_3_7/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/10000/7EC7DD7F-782C-E811-B469-0CC47A4D76A0.root'),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    inputCommands = cms.untracked.vstring("keep *", 
        "drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT",
        "drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT",
        "drop l1tEMTFHit2016s_simEmtfDigis__HLT",
        "drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT",
        "drop l1tEMTFTrack2016s_simEmtfDigis__HLT")

)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500))
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('L1Trigger.Phase2L1ParticleFlow.l1pfJetMetTreeProducer_cff')

process.p = cms.Path(process.l1pfJetMetTreeProducer)
process.TFileService = cms.Service("TFileService", fileName = cms.string("jetmetTuple.root"))
