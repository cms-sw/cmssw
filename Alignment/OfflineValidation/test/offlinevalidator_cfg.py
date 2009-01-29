import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

##
## Message Logger
##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100) # every 100th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))
process.MessageLogger.statistics.append('cout')

##
## Process options
##
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring("ProductNotFound") # make this exception fatal
    )

##
## Conditions
##
#process.load("Configuration.StandardSequences.FakeConditions_cff")
#process.load("CalibTracker.Configuration.Tracker_FakeConditions_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = 'IDEAL_30X::All' # 'CRUZET4_V3P::All'

##
## Geometry
##
process.load("Configuration.StandardSequences.Geometry_cff")

##
## Magnetic Field
##
process.load("Configuration.StandardSequences.MagneticField_cff")
# 0 T:
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")

##
## BeamSpot from database (i.e. GlobalTag), needed for Refitter
##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

##
## Load DBSetup (if needed)
##
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
process.trackerAlignment = poolDBESSource.clone(
    connect = 'frontier://FrontierProd/CMS_COND_21X_ALIGNMENT', # or your sqlite file
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerIdealGeometry210_mc') # your tag
        )
      )
    )
process.prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")

# APE always zero:
process.myTrackerAlignmentErr = poolDBESSource.clone(
    connect = cms.string('frontier://FrontierProd/CMS_COND_21X_ALIGNMENT'),
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('TrackerIdealGeometryErrors210_mc')
        )
      )
    )
process.es_prefer_trackerAlignmentErr = cms.ESPrefer("PoolDBESSource","myTrackerAlignmentErr")

##
## Input File(s)
##
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    ##
    ## 218 RelVal Sample, please replace accordingly
    ##
    '/store/relval/CMSSW_2_1_8/RelValZMM/ALCARECO/STARTUP_V7_StreamALCARECOTkAlMuonIsolated_v1/0003/A8583C5E-0283-DD11-8D18-000423D987FC.root'
#    ##
    ## CRUZET 4 ALCARECO (on CAF)
    ##
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/000690AE-E771-DD11-93D6-000423D996C8.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/002C355B-3372-DD11-A93C-000423D98DD4.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/00FD0190-7A71-DD11-A95F-000423D98844.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/0237113C-1F71-DD11-95B9-000423D8FA38.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/024501B6-CE71-DD11-88F6-000423D98930.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/043D4A5B-1E71-DD11-8C00-001D09F2441B.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/048E64E3-9371-DD11-8C2C-001D09F2AF1E.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/04E932C2-A471-DD11-8CEC-001D09F2924F.root',
#    '/store/data/Commissioning08/Cosmics/ALCARECO/CRUZET4_V2_StreamTkAlCosmics0T_CRUZET4_v2/0006/06075384-6171-DD11-975E-001617C3B65A.root' #,...
    )
)

##
## Maximum number of Events
##
process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
)

##
## Output File Configuration
##
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('AlignmentValidation.root')
)

##
## Load and Configure track selection for alignment
##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
# process.AlignmentTrackSelector.src = "ALCARECOTkAlCosmicsCTF0T" #e.g. for cosmics ALCARECO
process.AlignmentTrackSelector.src = "ALCARECOTkAlMuonIsolated" #e.g. for cosmics ALCARECO
process.AlignmentTrackSelector.applyBasicCuts = False #True
#process.AlignmentTrackSelector.ptMin   = .3

##
## Load and Configure TrackRefitter
##
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True

##
## Load and Configure OfflineValidation
##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_cfi")
process.TrackerOfflineValidation.Tracks = 'TrackRefitter'
process.TrackerOfflineValidation.trajectoryInput = 'TrackRefitter'
process.TrackerOfflineValidation.moduleLevelHistsTransient = True
process.TrackerOfflineValidation.TH1XprimeResStripModules.xmin = -0.2
process.TrackerOfflineValidation.TH1XprimeResStripModules.Nbinx = 100
process.TrackerOfflineValidation.TH1XprimeResStripModules.xmax = 0.2
#
process.TrackerOfflineValidation.TH1XprimeResPixelModules.xmin = -0.1
process.TrackerOfflineValidation.TH1XprimeResPixelModules.Nbinx = 100
process.TrackerOfflineValidation.TH1XprimeResPixelModules.xmax = 0.1
# Other used binnings you might want to replace:
#process.TrackerOfflineValidation.TH1YResPixelModules
#process.TrackerOfflineValidation.TH1NormYResPixelModules
#process.TrackerOfflineValidation.TH1NormXprimeResPixelModules
#process.TrackerOfflineValidation.TH1XResPixelModules
#process.TrackerOfflineValidation.TH1NormXResPixelModules
#process.TrackerOfflineValidation.TH1YResStripModules
#process.TrackerOfflineValidation.TH1NormYResStripModules
#process.TrackerOfflineValidation.TH1NormXprimeResStripModules
#process.TrackerOfflineValidation.TH1XResStripModules
#process.TrackerOfflineValidation.TH1NormXResStripModules

##
## PATH
##
process.p = cms.Path(process.offlineBeamSpot
                     *process.AlignmentTrackSelector
                     *process.TrackRefitter
                     *process.TrackerOfflineValidation
                     )
