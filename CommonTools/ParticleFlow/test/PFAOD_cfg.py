import FWCore.ParameterSet.Config as cms

process = cms.Process("PFAOD")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_4_2_0_pre8/RelValTTbar/GEN-SIM-RECO/START42_V7-v1/0042/4C740530-8B56-E011-8CCE-0018F3D095FC.root'
    )
)



process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))
#WARNING!
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.EventContent.EventContent_cff")
process.out = cms.OutputModule(
    "PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('PFAOD.root'),
    )

process.load("CommonTools.ParticleFlow.PF2PAT_EventContent_cff")
process.out.outputCommands.extend( process.prunedAODForPF2PATEventContent.outputCommands )

process.endpath = cms.EndPath(
    process.out
    )


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10


