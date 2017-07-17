import FWCore.ParameterSet.Config as cms

process = cms.Process("splitter")

# CMSSW.2.2.3

# message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_Offline_IdealGeom', 
        'cout')
)
## report only every 100th record
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

# needed for geometry
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

# magnetic field
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.Geometry.GeometryDB_cff")

# including global tag
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.load("Alignment.OfflineValidation.GlobalTag_cff")
# setting global tag
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_ALL_V4::All"
#process.GlobalTag.globaltag = "IDEAL_V9::All"
#process.GlobalTag.globaltag = "STARTUP_V8::All"

# track selectors and refitting
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:blah.root')
)

# including data...include your favorite data here
process.load("Alignment.OfflineValidation.CraftRepro_SkimAB_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(250000)
)


# adding geometries
from CondCore.DBCommon.CondDBSetup_cfi import *

# for craft

# CRAFT REPRO geom
process.trackerAlignment = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('Alignments')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/ntran/craftRepro/CRAFTrepro_HIP.db')
	)

# APEs from sqlite
process.ZeroAPE = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentErrorExtendedRcd'),
        tag = cms.string('TrackerSurveyLASOnlyErrors_def_210_mc')
		#tag = cms.string('TrackerCRAFTReRecoErrors_v1.10a_offline')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/zguo/APEtest/Alignments_APE_def.db')
)


# set prefer
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")

#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

# hit filter
process.load("Alignment.TrackHitFilter.TrackHitFilter_cfi")
# parameters for TrackHitFilter
#process.TrackHitFilter.src = "cosmictrackfinderP5"
#process.TrackHitFilter.src = 'ALCARECOTkAlCosmicsCTF'
process.TrackHitFilter.src = 'ALCARECOTkAlCosmicsCTF0T'

process.TrackHitFilter.hitSelection = "All"        
#process.TrackHitFilter.rejectBadStoNHits = True
#process.TrackHitFilter.theStoNthreshold = 14
#process.TrackHitFilter.minHitsForRefit = 6
#process.TrackHitFilter.rejectBadMods = True



# refit tracks first
process.TrackRefitterP5.src = 'TrackHitFilter'
#process.TrackRefitterP5.src = "ALCARECOTkAlCosmicsCosmicTF0T"
process.TrackRefitterP5.TrajectoryInEvent = True
process.TrackRefitterP5.TTRHBuilder = "WithTrackAngle"
process.FittingSmootherRKP5.EstimateCut = -1

# module configuration
# alignment track selector
#process.AlignmentTrackSelector.src = "ALCARECOTkAlCosmicsCTF0T"
process.AlignmentTrackSelector.src = "TrackRefitterP5"
#process.AlignmentTrackSelector.src = "TrackHitFilter"
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.pMin   = 4.	
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
process.AlignmentTrackSelector.minHitsPerSubDet.inBPIX = 2
process.KFFittingSmootherWithOutliersRejectionAndRK.EstimateCut=30.0
process.KFFittingSmootherWithOutliersRejectionAndRK.MinNumberOfHits=4



# configuration of the track spitting module
# new cuts allow for cutting on the impact parameter of the original track
process.load("RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi")
process.cosmicTrackSplitter.tracks='AlignmentTrackSelector'
process.cosmicTrackSplitter.tjTkAssociationMapTag='TrackRefitterP5'

#---------------------------------------------------------------------
# the output of the track hit filter are track candidates
# give them to the TrackProducer
process.ctfWithMaterialTracksP5.src = 'cosmicTrackSplitter'
process.ctfWithMaterialTracksP5.TrajectoryInEvent = True
process.ctfWithMaterialTracksP5.TTRHBuilder = "WithTrackAngle"

# second refit
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter2 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.TrackRefitter2.src = 'ctfWithMaterialTracksP5'
process.TrackRefitter2.TrajectoryInEvent = True
process.TrackRefitter2.TTRHBuilder = "WithTrackAngle"

process.cosmicValidation = cms.EDAnalyzer("CosmicSplitterValidation",
                                        ifSplitMuons = cms.bool(False),
                                        checkIfGolden = cms.bool(False),
                                        splitTracks = cms.InputTag("TrackRefitter2","","splitter"),
                                        splitGlobalMuons = cms.InputTag("muons","","splitter"),
                                        originalTracks = cms.InputTag("TrackRefitterP5"),
                                        originalGlobalMuons = cms.InputTag("muons","","Rec")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('output.root')
)

process.p = cms.Path(process.offlineBeamSpot*process.TrackHitFilter*process.TrackRefitterP5*process.AlignmentTrackSelector*process.cosmicTrackSplitter*process.ctfWithMaterialTracksP5*process.TrackRefitter2*process.cosmicValidation)


