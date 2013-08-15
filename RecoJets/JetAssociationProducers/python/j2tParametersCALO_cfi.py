import FWCore.ParameterSet.Config as cms

j2tParametersCALO = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    trackQuality = cms.string("goodIterative"),
    extrapolations = cms.InputTag("trackExtrapolator"),
    coneSize = cms.double(0.4)
)

