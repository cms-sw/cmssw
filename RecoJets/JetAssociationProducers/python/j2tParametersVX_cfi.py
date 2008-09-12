import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersVX.cfi,v 1.2 2008/02/19 21:43:07 fedor Exp $
j2tParametersVX = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

