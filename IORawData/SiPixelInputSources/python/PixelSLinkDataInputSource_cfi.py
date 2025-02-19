import FWCore.ParameterSet.Config as cms

source = cms.Source("PixelSLinkDataInputSource",
    runNumber = cms.untracked.int32(-1),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/PixelAlive_070106d.dmp'),
    fedid = cms.untracked.int32(-1)
)


