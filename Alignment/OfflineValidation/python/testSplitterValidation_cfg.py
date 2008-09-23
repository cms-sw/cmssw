import FWCore.ParameterSet.Config as cms

process = cms.Process("splitter")

# message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_Offline_IdealGeom', 
        'cout')
)
## report only every 100th record
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


# needed for geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

# including global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
# setting global tag
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V2::All"


# track selectors and refitting
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:///?svcClass=cmscaf&path=/castor/cern.ch/cms/store/cmscaf/alca/alignment/iCSA08/TkCosmicBOFF/ALCARECO/DATA/Run50908_0.root')
)

# including data...
#process.load("Alignment.OfflineValidation.CentralProd_330k_Splitting_cff")
process.load("Alignment.OfflineValidation.CentralProd_Cruzet4_V2P_interimReco_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000)
)


# magnetic field
process.load("MagneticField.Engine.uniformMagneticField_cfi")
process.UniformMagneticFieldESProducer.ZFieldInTesla = 0.0
#process.prefer("UniformMagneticFieldESProducer")

# adding geometries
from CondCore.DBCommon.CondDBSetup_cfi import *
# for ideal geometry
"""
process.trackerAlignment = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerIdealGeometry210_mc')
    )),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_21X_ALIGNMENT')
)
"""
#for cruzet4 geometry
process.trackerAlignment = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('Alignments')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/CRUZET4_DBobjects/alignments_C4fixPXESurveyV3.db')
)


process.ZeroAPE = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerIdealGeometryErrors210_mc')
    )),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_21X_ALIGNMENT')
)
# set prefer
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")


# module configuration
# alignment track selector
#process.AlignmentTrackSelector.src = "ALCARECOTkAlCosmicsCTF0T"
process.AlignmentTrackSelector.src = "cosmictrackfinderP5"
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.etaMin  = -9999.
process.AlignmentTrackSelector.etaMax  = 9999.
process.AlignmentTrackSelector.nHitMin = 6
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 100.
process.AlignmentTrackSelector.applyMultiplicityFilter = True
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.

process.TrackRefitter.src = "AlignmentTrackSelector"
process.TrackRefitter.TrajectoryInEvent = True

process.cosmicTrackSplitter = cms.EDFilter("CosmicTrackSplitter",
	stripFrontInvalidHits = cms.bool(True),
    stripBackInvalidHits = cms.bool(True),
    stripAllInvalidHits = cms.bool(False),
    replaceWithInactiveHits = cms.bool(True),
    tracks = cms.InputTag("TrackRefitter"),
    minimumHits = cms.uint32(6),
    detsToIgnore = cms.vuint32()
)

#---------------------------------------------------------------------
# the output of the track hit filter are track candidates
# give them to the TrackProducer
process.ctfWithMaterialTracksP5.src = 'cosmicTrackSplitter'
process.ctfWithMaterialTracksP5.TrajectoryInEvent = True

# if using refitter, but NOT
#process.ctfWithMaterialTracks.src = 'cosmicTrackSplitter'
#process.TrackRefitter.src = 'ctfWithMaterialTracks'
#process.TrackRefitter.TrajectoryInEvent = True
#process.TrackRefitter.TTRHBuilder = 'WithoutRefit'
#process.ttrhbwor.Matcher = 'StandardMatcher'


process.cosmicValidation = cms.EDFilter("CosmicSplitterValidation",
    tracks = cms.InputTag("ctfWithMaterialTracksP5")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('cosmicSplitterValidation.root')
)

process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.TrackRefitter*process.cosmicTrackSplitter*process.ctfWithMaterialTracksP5*process.cosmicValidation)
#process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.cosmicTrackSplitter*process.ctfWithMaterialTracks*process.TrackRefitter*process.cosmicValidation)

