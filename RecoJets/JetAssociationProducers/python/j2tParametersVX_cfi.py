import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersVX_cfi.py,v 1.5 2010/03/18 09:17:24 srappocc Exp $
j2tParametersVX = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5),
    useAssigned = cms.bool(False),
    pvSrc = cms.InputTag("offlinePrimaryVertices")
)

