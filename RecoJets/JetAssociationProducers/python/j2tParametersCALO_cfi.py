import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersCALO_cfi.py,v 1.5 2010/03/16 21:45:55 srappocc Exp $
j2tParametersCALO = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    trackQuality = cms.string("goodIterative"),
    extrapolations = cms.InputTag("trackExtrapolator"),
    coneSize = cms.double(0.5)
)

