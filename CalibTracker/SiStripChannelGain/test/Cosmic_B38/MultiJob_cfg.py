import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTI")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
      XXX_INPUT_XXX
  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(XXX_NEVENTS_XXX)
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

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V4P::All"
process.prefer("GlobalTag")

process.TrackRefitter.src = 'ctfWithMaterialTracksP5'
process.TrackRefitter.TrajectoryInEvent = True

process.SiStripCalib = cms.EDFilter("SiStripGainFromData",
    AlgoMode            = cms.string('MultiJob'),

    OutputHistos        = cms.string('XXX_OUTPUT_XXX.root'),
    OutputGains         = cms.string('XXX_OUTPUT_XXX.txt'),

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

