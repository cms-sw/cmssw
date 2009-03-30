import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersCALO_cfi.py,v 1.3 2008/05/29 17:58:55 fedor Exp $
j2tParametersCALO = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    trackQuality = cms.string("goodIterative"),
    coneSize = cms.double(0.5)
)

