import FWCore.ParameterSet.Config as cms

FP420Track = cms.EDProducer("TrackerizerFP420",
    ROUList = cms.vstring('FP420Cluster'),
    VerbosityLevel = cms.untracked.int32(0),
    NumberFP420Stations = cms.int32(3),
    NumberFP420Detectors = cms.int32(3),
    NumberFP420SPlanes = cms.int32(6),
    NumberFP420SPTypes = cms.int32(2),
    z420 = cms.double(420000.0),
    zD3 = cms.double(8000.0),
    zD2 = cms.double(4000.0),
    TrackModeFP420 = cms.string('TrackProducerSophisticatedFP420'),
    dXXFP420 = cms.double(4.7),
    dYYFP420 = cms.double(3.6),
    chiCutY420 = cms.double(3.0),
    chiCutX420 = cms.double(3.0)
)



