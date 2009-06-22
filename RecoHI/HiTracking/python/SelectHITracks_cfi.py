import FWCore.ParameterSet.Config as cms

# reco track quality cuts (dca, nhits, prob) and minpt selection
selectHiTracks = cms.EDFilter("TrackSelector",
  src = cms.InputTag("globalPrimTracks"),
  cut = cms.string('pt > 0.9 && d0/d0Error<3 && recHitsSize>12 && chi2prob(chi2,ndof)>0.01')
)
