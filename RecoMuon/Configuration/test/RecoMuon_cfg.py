import FWCore.ParameterSet.Config as cms

process = cms.Process("RecoMuon")
# Messages
#process.load("RecoMuon.Configuration.MessageLogger_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

# Muon Reco
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

process.load("RecoMuon.Configuration.RecoMuon_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:muPt10_1.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('RecoMuons.root')
)

process.p = cms.Path(process.RawToDigi*process.trackerlocalreco*process.ckftracks*process.muonreco_plus_isolation*process.muoncosmicreco)
process.this_is_the_end = cms.EndPath(process.out)

process.GlobalTag.globaltag = 'IDEAL_V9::All'


