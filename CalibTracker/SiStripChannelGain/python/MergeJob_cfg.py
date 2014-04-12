# The following comments couldn't be translated into the new config version:

# XXX_SKIPEVENT_XXX

#string timetype = "timestamp"    

import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoTracker.TrackProducer.TrackRefitter_cff")

process.source = cms.Source("EmptyIOVSource",
    lastRun = cms.untracked.uint32(2),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    suppressWarning = cms.untracked.vstring('TrackRefitter'),
    suppressInfo = cms.untracked.vstring('TrackRefitter'),
    suppressDebug = cms.untracked.vstring('TrackRefitter'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.GlobalTag.globaltag = 'STARTUP_V7::All'

process.TrackRefitter.src = 'ALCARECOSiStripCalMinBias'
process.TrackRefitter.TrajectoryInEvent = True


process.SiStripCalib = cms.EDFilter("SiStripGainFromData",
    AlgoMode            = cms.string('WriteOnDB'),

    VInputFiles         = cms.vstring('file:out.root'),

    OutputHistos        = cms.string('SiStripCalib.root'),
    OutputGains         = cms.string('SiStripCalib.txt'),

    TrajToTrackProducer = cms.string('TrackRefitter'),
    TrajToTrackLabel    = cms.string(''),

    minTrackMomentum    = cms.untracked.double(1.0),
    minNrEntries        = cms.untracked.uint32(100),
    maxChi2OverNDF      = cms.untracked.double(9999999.0),
    maxMPVError         = cms.untracked.double(1000000.0),
    maxNrStrips         = cms.untracked.uint32(8),

    FirstSetOfConstants = cms.untracked.bool(False),

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
    connect = cms.string('sqlite_file:SiStrip_ChannelGain_MinBias_iCSA08_S156.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStrip_Gain_MinBias_CSA08_s156_mc')
    ))
)

process.p = cms.Path(process.SiStripCalib)

