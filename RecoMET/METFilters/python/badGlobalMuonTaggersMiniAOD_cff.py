import FWCore.ParameterSet.Config as cms

badGlobalMuonTagger = cms.EDFilter("BadGlobalMuonTagger",
    muons = cms.InputTag("slimmedMuons"),
    vtx   = cms.InputTag("offlineSlimmedPrimaryVertices"),
    muonPtCut = cms.double(20),
    selectClones = cms.bool(False),
    taggingMode = cms.bool(False),
)
cloneGlobalMuonTagger = badGlobalMuonTagger.clone(
    selectClones = True
)

noBadGlobalMuons = cms.Sequence(~cloneGlobalMuonTagger + ~badGlobalMuonTagger)
    
