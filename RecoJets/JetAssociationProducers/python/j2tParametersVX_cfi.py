import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersVX_cfi.py,v 1.2 2008/04/21 03:27:47 rpw Exp $
j2tParametersVX = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    trackQuality = "goodIterative",
    coneSize = cms.double(0.5)
)

