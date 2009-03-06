import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersCALO.cfi,v 1.2 2008/02/19 21:43:07 fedor Exp $
j2tParametersCALO = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

