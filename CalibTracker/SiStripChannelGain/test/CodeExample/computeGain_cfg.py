# The following comments couldn't be translated into the new config version:

# XXX_SKIPEVENT_XXX

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(110219),
    lastValue  = cms.uint64(111129)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('ERROR')
    )
)

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V6P::All"
process.prefer("GlobalTag")

process.SiStripCalib = cms.EDFilter("SiStripGainFromCalibTree",
    OutputGains         = cms.string('XXX_LOCALPATH_XXX/Gains_ASCI.txt'),

    minTrackMomentum    = cms.untracked.double(1),
    minNrEntries        = cms.untracked.double(10),
    maxChi2OverNDF      = cms.untracked.double(9999999.0),
    maxMPVError         = cms.untracked.double(25.0),
    maxNrStrips         = cms.untracked.uint32(8),

    Validation          = cms.untracked.bool(False),
    OldGainRemoving     = cms.untracked.bool(False),
    FirstSetOfConstants = cms.untracked.bool(True),

    CalibrationLevel    = cms.untracked.int32(2), # 0==APV, 1==Laser, 2==module

    InputFiles          = cms.vstring(
        "rfio:/castor/cern.ch/user/k/kaschube/calibration/calibTree_run123592.root",
        "rfio:/castor/cern.ch/user/k/kaschube/calibration/calibTree_run123596.root",
        "rfio:/castor/cern.ch/user/k/kaschube/calibration/calibTree_run123615.root",
        "rfio:/castor/cern.ch/user/k/kaschube/calibration/calibTree_run123732.root"
    ),


    SinceAppendMode     = cms.bool(True),
    IOVMode             = cms.string('Job'),
    Record              = cms.string('SiStripApvGainRcd'),
    doStoreOnDB         = cms.bool(True)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:XXX_LOCALPATH_XXX/Gains_Sqlite.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SST_Gain_CRAFT09_Deco')
    ))
)

#process.load("DQM.SiStripMonitorClient.SiStripDQMOnline_cff")

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('XXX_LOCALPATH_XXX/Gains_Tree.root')  
)

#process.p = cms.Path(process.SiStripCalib*process.dqmSaver)
process.p = cms.Path(process.SiStripCalib)

