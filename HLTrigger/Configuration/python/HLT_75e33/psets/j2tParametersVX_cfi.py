import FWCore.ParameterSet.Config as cms

j2tParametersVX = cms.PSet(
    coneSize = cms.double(0.4),
    pvSrc = cms.InputTag("offlinePrimaryVertices"),
    tracks = cms.InputTag("generalTracks"),
    useAssigned = cms.bool(False)
)