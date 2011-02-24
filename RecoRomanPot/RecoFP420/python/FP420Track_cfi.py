import FWCore.ParameterSet.Config as cms

FP420Track = cms.EDProducer("TrackerizerFP420",
    ROUList = cms.vstring('FP420Cluster'),
    VerbosityLevel = cms.untracked.int32(0),
    NumberFP420Detectors = cms.int32(3),        ## =3 means 2 Trackers: +FP420 and -FP420; =0 -> no FP420 at all 
    NumberFP420Stations = cms.int32(3),         ## means 2 Stations w/ arm 8m
    NumberFP420SPlanes = cms.int32(6),          ## means 5 SuperPlanes
    NumberFP420SPTypes = cms.int32(2),          ##    xytype
    zFP420 = cms.double(420000.0),              ##
    zFP420D3 = cms.double(8000.0),              ##
    zFP420D2 = cms.double(4000.0),              ##
    TrackModeFP420 = cms.string('TrackProducerSophisticatedFP420'), ##
    dXXFP420 = cms.double(4.7),                 ##
    dYYFP420 = cms.double(3.6),                 ##
    chiCutYFP420 = cms.double(3.0),             ##
    chiCutXFP420 = cms.double(3.0),             ##
    NumberHPS240Detectors = cms.int32(3),            ## =3 means 2 Trackers: +HPS240 and -HPS240; =0 -> no HPS240 at all 
    NumberHPS240Stations = cms.int32(3),             ## means 2 Stations w/ arm 8m
    NumberHPS240SPlanes = cms.int32(6),              ## means 5 SuperPlanes
    NumberHPS240SPTypes = cms.int32(2),              ##    xytype
    zHPS240 = cms.double(240000.0),                  ##
    zHPS240D3 = cms.double(8000.0),                  ##
    zHPS240D2 = cms.double(4000.0),                  ##
    TrackModeHPS240 = cms.string('TrackProducerSophisticatedHPS240'), ##
    dXXHPS240 = cms.double(2.0),                     ##
    dYYHPS240 = cms.double(3.6),                     ##
    chiCutYHPS240 = cms.double(3.0),                 ##
    chiCutXHPS240 = cms.double(3.0)                  ##
)



