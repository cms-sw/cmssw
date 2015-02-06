import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

####################################################################
## Message Logger
####################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1000) # every 100th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))


process.MessageLogger.statistics.append('cout')
process.options = cms.untracked.PSet(
        Rethrow = cms.untracked.vstring("ProductNotFound") # make this exception fatal
        #    , fileMode  =  cms.untracked.string('FULLMERGE') # any file order (default): caches all lumi/run products (memory!)
        #    , fileMode  =  cms.untracked.string('MERGE') # needs files sorted in run and within run in lumi sections (hard to achieve)
            , fileMode  =  cms.untracked.string('NOMERGE') # needs files sorted in run, caches lumi
            )
#process.options = cms.untracked.PSet(
#    Rethrow = cms.untracked.vstring("ProductNotFound") # make this exception fatal
#    )

#########################################################################
## Conditions
#########################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = "MC_31X_V9::All"


##################################################################
## Geometry
##################################################################
process.load("Configuration.Geometry.GeometryDB_cff")


###############################################################
## Magnetic Field
###############################################################

process.load("Configuration.StandardSequences.MagneticField_cff")
# for 0 T:
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")

#####################################################################
## BeamSpot from database (i.e. GlobalTag), needed for Refitter
#####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

#####################################################################
## Load DBSetup (if needed)
####################################################################
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
##
##include private db object
##
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.myTrackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
    connect = cms.string ('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
    toGet = cms.VPSet(cms.PSet(
    record = cms.string('TrackerAlignmentRcd'),
#    tag = cms.string('TrackerIdealGeometry210_mc')#IDEAL
    tag = cms.string('Tracker_Geometry_v5_offline')#SHMm
  ))
)
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","myTrackerAlignment")

############################################################################
# APE always zero:
###########################################################################
process.myTrackerAlignmentErr = poolDBESSource.clone(
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
    toGet = cms.VPSet(
      cms.PSet(
       record = cms.string('TrackerAlignmentErrorExtendedRcd'),
        tag = cms.string('TrackerIdealGeometryErrors210_mc')
       )
      )
   )
process.es_prefer_trackerAlignmentErr = cms.ESPrefer("PoolDBESSource","myTrackerAlignmentErr")

#######################################################################
## Input File(s)
#######################################################################
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/E640526B-3AE2-DE11-AAC3-000423D94990.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/CED18FDD-34E2-DE11-BA11-000423D9A2AE.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/CC81BA78-4DE2-DE11-8988-000423D6A6F4.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/CA9B4219-3AE2-DE11-802D-0019B9F72CE5.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/BAF9885B-3BE2-DE11-8913-001D09F29114.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/BA4F0900-39E2-DE11-ABC7-000423D9A2AE.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/B27878C8-49E2-DE11-807A-001D09F2915A.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/9AF18106-42E2-DE11-83CB-001D09F27067.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/9453CF7A-42E2-DE11-87D6-0019B9F72BAA.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/8C3A3DD9-32E2-DE11-80F9-001617E30D12.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/848CB8C6-3EE2-DE11-8622-001D09F241B9.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/82A4D042-3FE2-DE11-93FA-001617C3B65A.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/6A9C296C-4BE2-DE11-902C-000423D99614.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/5039D8E0-36E2-DE11-BDCA-000423D987FC.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/44B26111-47E2-DE11-9D27-001D09F28D54.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/32BB8713-3CE2-DE11-9DD6-001617C3B79A.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/26B6B47B-44E2-DE11-9C12-001D09F2545B.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/18454E8A-43E2-DE11-87F6-0030486780B8.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/0C1678BF-47E2-DE11-854A-0019B9F72D71.root',
        '/store/express/BeamCommissioning09/StreamExpress/ALCARECO/v2/000/123/596/02E3D234-37E2-DE11-8E73-001617C3B5F4.root'
    )
)
#####################################################################
## Input source manipulation
###################################################################
## set first run
#process.source.firstRun = cms.untracked.uint32(67534)
#process.source.lastRun = cms.untracked.uint32(67647)
## Maximum number of Events
process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1001)
)

## select sped
#process.load("AuxCode.RunNumberFilter.RunNumberFilter_cfi")
#process.RunNumberFilter.doRunSelection = True
#process.RunNumberFilter.selectedRunNumber = 66748


###################################################################
## Load and Configure TrackRefitter
##First refit with applying geometry set above
##Second refit after the track selection
###################################################################
# refitting
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter1 = process.TrackRefitter.clone(
    src = 'ALCARECOTkAlMinBias',
    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate", #"WithTrackAngle"
    NavigationSchool = ''
)

process.TrackRefitterForOfflineValidation = process.TrackRefitter1.clone(
    src = 'AlignmentTrackSelector',
)

#######################################################################
##TRACKER TRACK HIT FILTER
##First remove 'bad' hits from track
##Second run track producer on filtered hits
######################################################################
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
      # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 14.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

################################################################################################
#TRACK PRODUCER
#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
################################################################################################
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff   #TrackRefitters_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(
    src = 'TrackerTrackHitFilter',
    #TrajectoryInEvent = True
    TTRHBuilder = "WithAngleAndTemplate"
)



###############################################################
## Load and Configure track selection for alignment
###############################################################
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src ='HitFilteredTracks' #'TrackRefitter1'
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 1.5
process.AlignmentTrackSelector.pMin   = 0.
process.AlignmentTrackSelector.nHitMin =10
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 100.
#process.AlignmentTrackSelector.minHitsPerSubDet.inBPIX = 2

#process.AlignmentTrackSelector.nHitsMinLowerPart = 0


###################################################################
## Load and Configure OfflineValidation and Output File
##################################################################
#Offline Validation Parameters
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_Standalone_cff")
#process.TrackerOfflineValidationStandalone.Tracks = 'TrackRefitter2'
#process.TrackerOfflineValidationStandalone.trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidationStandalone.moduleLevelHistsTransient = True #False
# Output File
process.TFileService.fileName = '$TMPDIR/Validation_output.root'



process.dump=cms.EDAnalyzer("EventContentAnalyzer")


##############################################################################
## PATH
##############################################################################
process.p = cms.Path(process.offlineBeamSpot
                     *process.TrackRefitter1
                     *process.TrackerTrackHitFilter
                     *process.HitFilteredTracks 
                     *process.AlignmentTrackSelector
                     *process.TrackRefitterForOfflineValidation
                     *process.seqTrackerOfflineValidationStandalone
                     )
