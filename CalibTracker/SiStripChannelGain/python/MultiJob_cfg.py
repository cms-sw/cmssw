import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTI")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoTracker.TrackProducer.TrackRefitter_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_6/RelValMinBias/ALCARECO/STARTUP_V7_StreamALCARECOSiStripCalMinBias_v2/0001/34BE0343-B979-DD11-9E9E-001617DC1F70.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    suppressWarning   = cms.untracked.vstring('TrackRefitter',         'FwkReport'),
    suppressInfo      = cms.untracked.vstring('TrackRefitter'   ,         'FwkReport'),
    suppressDebug     = cms.untracked.vstring('TrackRefitter'  ,         'FwkReport'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.GlobalTag.globaltag = 'STARTUP_V7::All'

process.TrackRefitter.src = 'ALCARECOSiStripCalMinBias'
process.TrackRefitter.TrajectoryInEvent = True

process.SiStripCalib = cms.EDFilter("SiStripGainFromData",
    AlgoMode            = cms.string('MultiJob'),

    OutputHistos        = cms.string('out.root'),
    OutputGains         = cms.string('out.txt'),

    TrajToTrackProducer = cms.string('TrackRefitter'),
    TrajToTrackLabel    = cms.string(''),

    minTrackMomentum    = cms.untracked.double(1.0),
    minNrEntries        = cms.untracked.uint32(100),
    maxChi2OverNDF      = cms.untracked.double(9999999.0),
    maxMPVError         = cms.untracked.double(1000000.0),
    maxNrStrips         = cms.untracked.uint32(8),

    SinceAppendMode     = cms.bool(True),
    IOVMode             = cms.string('Job'),
    Record              = cms.string('SiStripApvGainRcd'),
    doStoreOnDB         = cms.bool(False)
)

process.p = cms.Path(process.TrackRefitter*process.SiStripCalib)

