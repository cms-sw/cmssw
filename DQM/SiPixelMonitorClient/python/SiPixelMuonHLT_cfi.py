import FWCore.ParameterSet.Config as cms

siPixelMuonHLT = cms.EDAnalyzer("SiPixelMuonHLT",
    outputFile = cms.untracked.string('./PixelMuonHLTDQM.root'),
    verbose    = cms.untracked.bool(False),
    monitorName = cms.untracked.string("HLT/HLTMonMuon"),
    saveOUTput  = cms.untracked.bool(False),
    clusterCollectionTag = cms.untracked.InputTag("hltSiPixelClusters"),
    rechitsCollectionTag = cms.untracked.InputTag("hltSiPixelRecHits"),
    l3MuonCollectionTag  = cms.untracked.InputTag("hltL3MuonCandidates")
)
