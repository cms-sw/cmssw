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
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.load("Alignment.OfflineValidation.GlobalTag_cff")
# setting global tag
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"


# track selectors and refitting
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:///?svcClass=cmscaf&path=/castor/cern.ch/cms/store/cmscaf/alca/alignment/iCSA08/TkCosmicBOFF/ALCARECO/DATA/Run50908_0.root')
)


# including data...
process.load("Alignment.OfflineValidation.CraftALCARECO_v7_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)


# magnetic field
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# adding geometries
from CondCore.DBCommon.CondDBSetup_cfi import *

# for craft

process.trackerAlignment = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('Alignments')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/ntran/CRAFT_output/craft1300k_pxbDetswSC_inflErrs_15iters/alignments.db')
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
"""

"""
# for ideal geometry

process.trackerAlignment = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerIdealGeometry210_mc')
    )),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_21X_ALIGNMENT')
)
"""
"""
process.ZeroAPE = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerIdealGeometryErrors210_mc')
    )),
    connect = cms.string('frontier://cms_conditions_data/CMS_COND_21X_ALIGNMENT')
)
"""

# set prefer
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
#process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")

# refit tracks first
#process.TrackRefitterP5.src = "cosmictrackfinderP5"
process.TrackRefitterP5.src = "ALCARECOTkAlCosmicsCosmicTF0T"
process.TrackRefitterP5.TrajectoryInEvent = True
process.TrackRefitterP5.TTRHBuilder = "WithTrackAngle"
process.FittingSmootherRKP5.EstimateCut = -1

# module configuration
# alignment track selector
#process.AlignmentTrackSelector.src = "ALCARECOTkAlCosmicsCTF0T"
process.AlignmentTrackSelector.src = "TrackRefitterP5"
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.pMin   = 5.	
process.AlignmentTrackSelector.ptMax   = 9999.	
process.AlignmentTrackSelector.pMax   = 9999.	
process.AlignmentTrackSelector.etaMin  = -9999.
process.AlignmentTrackSelector.etaMax  = 9999.
process.AlignmentTrackSelector.nHitMin = 10
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 9999.
process.AlignmentTrackSelector.applyMultiplicityFilter = True
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.


# configuration of the track spitting module
# new cuts allow for cutting on the impact parameter of the original track
process.cosmicTrackSplitter = cms.EDFilter("CosmicTrackSplitter",
	stripFrontInvalidHits = cms.bool(True),
    stripBackInvalidHits = cms.bool(True),
    stripAllInvalidHits = cms.bool(False),
    replaceWithInactiveHits = cms.bool(True),
    tracks = cms.InputTag("AlignmentTrackSelector"),
    minimumHits = cms.uint32(6),
    detsToIgnore = cms.vuint32(),
	dzCut = cms.double( 25.0 ),
	dxyCut = cms.double( 10.0 )
)

#---------------------------------------------------------------------
# the output of the track hit filter are track candidates
# give them to the TrackProducer
process.ctfWithMaterialTracksP5.src = 'cosmicTrackSplitter'
process.ctfWithMaterialTracksP5.TrajectoryInEvent = True
process.ctfWithMaterialTracksP5.TTRHBuilder = "WithTrackAngle"



process.cosmicValidation = cms.EDFilter("CosmicSplitterValidation",
    tracks = cms.InputTag("ctfWithMaterialTracksP5"),
	checkIfGolden = cms.bool(False)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('CRAFT500k/Craft500k_Craft_wPXBcut_noOR.root')
)

process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitterP5*process.AlignmentTrackSelector*process.cosmicTrackSplitter*process.ctfWithMaterialTracksP5*process.cosmicValidation)

