import FWCore.ParameterSet.Config as cms

# $Id: j2tParametersCALO_cfi.py,v 1.2 2008/04/21 03:27:46 rpw Exp $
j2tParametersCALO = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    trackQuality = "goodIterative",
    coneSize = cms.double(0.5)
)

