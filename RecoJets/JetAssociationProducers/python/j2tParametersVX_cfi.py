import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersVX_cfi.py,v 1.6 2011/11/11 18:58:01 srappocc Exp $
j2tParametersVX = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5),
    useAssigned = cms.bool(False),
    pvSrc = cms.InputTag("offlinePrimaryVertices")
)

