import FWCore.ParameterSet.Config as cms

PixelNtuplizer_RD = cms.EDFilter("PixelNtuplizer_RD",
    trajectoryInput = cms.string('TrackRefitter'),
    OutputFile = cms.string('TTreeFile.root')
)

