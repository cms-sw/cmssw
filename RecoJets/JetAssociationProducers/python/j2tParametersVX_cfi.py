import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersVX_cfi.py,v 1.4 2009/03/30 15:07:42 bainbrid Exp $
j2tParametersVX = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

