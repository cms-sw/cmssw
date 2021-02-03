import FWCore.ParameterSet.Config as cms

j2tParametersCALO = cms.PSet(
    coneSize = cms.double(0.4),
    extrapolations = cms.InputTag("trackExtrapolator"),
    trackQuality = cms.string('goodIterative'),
    tracks = cms.InputTag("generalTracks")
)