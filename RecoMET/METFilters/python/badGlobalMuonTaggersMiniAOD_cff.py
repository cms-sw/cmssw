import FWCore.ParameterSet.Config as cms

badGlobalMuonTaggerMAOD = cms.EDFilter("BadGlobalMuonTagger",
    muons = cms.InputTag("slimmedMuons"),
    vtx   = cms.InputTag("offlineSlimmedPrimaryVertices"),
    muonPtCut = cms.double(20),
    selectClones = cms.bool(False),
    taggingMode = cms.bool(False),
)
cloneGlobalMuonTaggerMAOD = badGlobalMuonTaggerMAOD.clone(
    selectClones = True
)

noBadGlobalMuonsMAOD = cms.Sequence(~cloneGlobalMuonTaggerMAOD + ~badGlobalMuonTaggerMAOD)
    
