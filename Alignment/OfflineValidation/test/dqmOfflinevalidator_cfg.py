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
    , fileMode  =  cms.untracked.string('NOMERGE')
)

##
## Conditions
##
#process.load("Configuration.StandardSequences.FakeConditions_cff")
#process.load("CalibTracker.Configuration.Tracker_FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = 'DESIGN_31X_V2::All' # 'CRUZET4_V3P::All'
#process.prefer("GlobalTag")

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
    connect = 'frontier://FrontierProd/CMS_COND_31X_FROM21X', # or your sqlite file
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('TrackerIdealGeometry210_mc') # your tags
        )
      )
    )
process.prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")

# APE always zero:
process.myTrackerAlignmentErr = poolDBESSource.clone(
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
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
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/065EC35D-00FF-DD11-A118-0018F3D0962C.root',
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/E282C810-01FF-DD11-A190-001731AF6B83.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/769DEEE1-00FF-DD11-8F8C-003048678D9A.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/3EB6FD4C-00FF-DD11-8E5F-0018F3D095EC.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/AC09161D-01FF-DD11-B9B5-00304875ABEF.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/96C579B9-00FF-DD11-AD81-0030486792B8.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/541A062C-03FF-DD11-9B1A-0018F3D0962A.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/EC9A5776-03FF-DD11-9F6B-0030486790A6.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/9281F6F7-00FF-DD11-954B-0030486791BA.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/D6DABE86-02FF-DD11-A18C-0018F3D0961E.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/6C41AD2C-03FF-DD11-9D51-001731AF6BCB.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/3EAC63B3-01FF-DD11-9BD9-0018F3D095FE.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/4676C0F2-01FF-DD11-95C3-0018F3D0960A.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/027D9E5B-02FF-DD11-B794-001731AF67B9.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/B697F214-03FF-DD11-9E7B-003048679168.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/5A2EC49F-02FF-DD11-B36A-003048678FA0.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/3A0298FF-01FF-DD11-8D17-001A92811700.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/601B9F26-03FF-DD11-BC24-003048679214.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/8ABAE8E6-02FF-DD11-8FFB-0018F3D096E8.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/50B25624-03FF-DD11-8C6F-0030486790C0.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/88210E11-03FF-DD11-816C-003048D15DB6.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/AE4CC64E-02FF-DD11-8C00-001A92810AF2.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/8C320055-03FF-DD11-9D44-001A92971B9A.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/46082956-03FF-DD11-80FC-00304875ABEF.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0005/D67A5CC5-81FE-DD11-958E-003048767D3D.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/949950CD-02FF-DD11-AF6E-001A92810AF2.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0008/CA986B41-02FF-DD11-9779-0030486792B8.root', 
    '/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V9_StreamALCARECOTkAlCosmics0T_225-v3/0007/2C42F11A-01FF-DD11-8A80-00304867902C.root'
    ##
    ## 218 RelVal Sample, please replace accordingly
    ##
####################    '/store/relval/CMSSW_2_1_8/RelValZMM/ALCARECO/STARTUP_V7_StreamALCARECOTkAlMuonIsolated_v1/0003/A8583C5E-0283-DD11-8D18-000423D987FC.root'
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
    input = cms.untracked.int32(101)
)

##
## Output File Configuration
##
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('/afs/cern.ch/user/h/hauk/scratch0/rootFiles/AlignmentValidation.root')
)

##
## Load and Configure track selection for alignment
##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = "ALCARECOTkAlCosmicsCTF0T" #e.g. for cosmics ALCARECO
####################process.AlignmentTrackSelector.src = "ALCARECOTkAlMuonIsolated" #e.g. for cosmics ALCARECO
process.AlignmentTrackSelector.applyBasicCuts = False #True
#process.AlignmentTrackSelector.ptMin   = .3

##
## Load and Configure TrackRefitter
##
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
#process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitterP5.src = 'AlignmentTrackSelector'
#process.TrackRefitter.TrajectoryInEvent = True # is default
process.TrackRefitterP5.TTRHBuilder = 'WithTrackAngle'
process.TrackRefitterP5.TrajectoryInEvent = True


##
## Load and Configure OfflineValidation
##
#process.load("Alignment.OfflineValidation.TrackerOfflineValidation_cfi")
#process.TrackerOfflineValidation.useInDqmMode = True
#process.TrackerOfflineValidation.Tracks = 'TrackRefitter'
#process.TrackerOfflineValidation.trajectoryInput = 'TrackRefitter'
##process.TrackerOfflineValidation.moduleLevelHistsTransient = True
#process.TrackerOfflineValidation.TH1XprimeResStripModules.xmin = -0.2
#process.TrackerOfflineValidation.TH1XprimeResStripModules.Nbinx = 100
#process.TrackerOfflineValidation.TH1XprimeResStripModules.xmax = 0.2
##
#process.TrackerOfflineValidation.TH1XprimeResPixelModules.xmin = -0.1
#process.TrackerOfflineValidation.TH1XprimeResPixelModules.Nbinx = 100
#process.TrackerOfflineValidation.TH1XprimeResPixelModules.xmax = 0.1
## Other used binnings you might want to replace:
##process.TrackerOfflineValidation.TH1YResPixelModules
##process.TrackerOfflineValidation.TH1NormYResPixelModules
##process.TrackerOfflineValidation.TH1NormXprimeResPixelModules
##process.TrackerOfflineValidation.TH1XResPixelModules
##process.TrackerOfflineValidation.TH1NormXResPixelModules
##process.TrackerOfflineValidation.TH1YResStripModules
##process.TrackerOfflineValidation.TH1NormYResStripModules
##process.TrackerOfflineValidation.TH1NormXprimeResStripModules
##process.TrackerOfflineValidation.TH1XResStripModules
##process.TrackerOfflineValidation.TH1NormXResStripModules

process.load("Alignment.OfflineValidation.TrackerOfflineValidation_Dqm_cff")
process.TrackerOfflineValidationDqm.Tracks = 'TrackRefitterP5'
process.TrackerOfflineValidationDqm.trajectoryInput = 'TrackRefitterP5'


process.load("Alignment.OfflineValidation.TrackerOfflineValidationSummary_cfi")


# DQM backend
process.load("DQMServices.Core.DQM_cfg")


# DQM file saver
process.dqmSaverMy = cms.EDFilter("DQMFileSaver",
          convention=cms.untracked.string("Offline"),
          workflow=cms.untracked.string("/Cosmics/TrackAlign_322patch2_R000100000_R000100050_v1/ALCARECO"),   # /primaryDatasetName/WorkflowDescription/DataTier; Current Convention: Indicate run range (first and last run) in file name 
          dirName=cms.untracked.string("/afs/cern.ch/user/h/hauk/scratch0/rootFiles/."),
	  #dirName=cms.untracked.string("."),
          saveByRun=cms.untracked.int32(-1),
	  saveAtJobEnd=cms.untracked.bool(True),                        
          forceRunNumber=cms.untracked.int32(100000)   # Current Convention: Take first processed run
)


##
## PATH
##
process.p = cms.Path(process.offlineBeamSpot
                     *process.AlignmentTrackSelector
                     #*process.TrackRefitter
                     *process.TrackRefitterP5
		     #*process.TrackerOfflineValidation
		     *process.TrackerOfflineValidationDqm
		     *process.TrackerOfflineValidationSummary
		     *process.dqmSaverMy
                     )
