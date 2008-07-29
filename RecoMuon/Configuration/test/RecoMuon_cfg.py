import FWCore.ParameterSet.Config as cms

process = cms.Process("RecoMuon")
# Messages
process.load("RecoMuon.Configuration.MessageLogger_cfi")

# Muon Reco
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

process.load("RecoMuon.Configuration.RecoMuon_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/bellan/trunkedGLB_210_pre11.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('RecoMuons.root')
)

process.p = cms.Path(process.muonreco)
process.this_is_the_end = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'IDEAL_V5::All'


