import FWCore.ParameterSet.Config as cms

FP420Track = cms.EDFilter("TrackerizerFP420",
    z420 = cms.double(420000.0),
    NumberFP420Stations = cms.int32(3),
    chiCutX420 = cms.double(50.0),
    #--------------------------------
    #--------------------------------
    ROUList = cms.vstring('FP420Cluster'),
    NumberFP420Detectors = cms.int32(3),
    NumberFP420SPTypes = cms.int32(2),
    zD3 = cms.double(8000.0),
    zD2 = cms.double(4000.0),
    chiCutY420 = cms.double(2.0),
    #--------------------------------
    #-----------------------------FP420TrackMain
    #--------------------------------
    TrackModeFP420 = cms.string('TrackProducerSophisticatedFP420'),
    #--------------------------------
    #-----------------------------TrackerizerFP420 
    #--------------------------------
    VerbosityLevel = cms.untracked.int32(0),
    dYYFP420 = cms.double(5.0),
    NumberFP420SPlanes = cms.int32(6),
    dXXFP420 = cms.double(4.7)
)


