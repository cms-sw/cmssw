import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersCALO_cfi.py,v 1.4 2009/03/30 15:07:42 bainbrid Exp $
j2tParametersCALO = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    trackQuality = cms.string("goodIterative"),
    extrapolations = cms.InputTag("trackExtrapolator"),
    coneSize = cms.double(0.5)
)

