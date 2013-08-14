import FWCore.ParameterSet.Config as cms

j2tParametersVX = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5),
    useAssigned = cms.bool(False),
    pvSrc = cms.InputTag("offlinePrimaryVertices")
)

