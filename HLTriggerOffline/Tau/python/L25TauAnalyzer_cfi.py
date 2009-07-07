import FWCore.ParameterSet.Config as cms
L25TauAnalyzer = cms.EDAnalyzer("L25TauAnalyzer",
    tauSource = cms.InputTag("pfTaus"),
    l25JetSource = cms.InputTag("hltL25TauPixelTracksConeIsolation"),
    l25PtCutSource = cms.InputTag("hltL25TauPixelTracksLeadingTrackPtCutSelector"),
    l25IsoSource = cms.InputTag("hltL25TauPixelTracksIsolationSelector"),
    matchingCone = cms.double(0.3),
)


