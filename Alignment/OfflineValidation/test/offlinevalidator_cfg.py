import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

##
## Message Logger
##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
## report only every 100th record
process.MessageLogger.cerr.FwkReport.reportEvery = 100

## 
## Database configuration
##
# process.load("CondCore.DBCommon.CondDBCommon_cfi")
# process.load("CondCore.DBCommon.CondDBSetup_cfi")

##
## Fake Conditions (if needed)
##
process.load("Configuration.StandardSequences.FakeConditions_cff")

##
## Geometry
##
process.load("Configuration.StandardSequences.Geometry_cff")

##
## Magnetic Field
##
process.load("Configuration.StandardSequences.MagneticField_cff")

##
## Load DBSetup (if needed)
##
from Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_DevDB_cff import *
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
trackerAlignment = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *

es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")

trackerAlignment.connect = 'frontier://FrontierDev/CMS_COND_ALIGNMENT'
trackerAlignment.toGet = cms.VPSet(cms.PSet(
    record = cms.string('TrackerAlignmentRcd'),
    tag = cms.string('Tracker10pbScenario210_mc')
), 
    cms.PSet(
        record = cms.string('TrackerAlignmentErrorRcd'),
        tag = cms.string('Tracker10pbScenarioErrors210_mc')
    ))
# to apply misalignments
TrackerDigiGeometryESModule.applyAlignment = True

##
## Input File(s)
##
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    ##
    ## 21X RelVal Samples, please replace accordingly
    ##
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/06D7B343-365C-DD11-AD13-001617E30D00.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/08056306-445C-DD11-A35C-000423D992A4.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/1C1712DA-365C-DD11-870A-001617C3B79A.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/1CA078CB-375C-DD11-96D2-001617E30D4A.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/2282BAF4-345C-DD11-83BF-001617C3B710.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/2CA50990-3B5C-DD11-8305-000423D6C8E6.root',
    '/store/relval/CMSSW_2_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/3014096D-375C-DD11-A2C0-001617E30F4C.root')

)

##
## Maximum number of Events
##
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

##
## Output File Configuration
##
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('AlignmentValidation_10pb_DB_ModOn.root')
)

##
## Load and Configure track selection for alignment
##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
#process.AlignmentTrackSelector.applyBasicCuts = True
#process.AlignmentTrackSelector.ptMin   = .3



##
## Load and Configure TrackRefitter
##
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True


##
## Load and Configure OfflineValidation
##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_cfi")
process.TrackerOfflineValidation.Tracks = 'TrackRefitter'
process.TrackerOfflineValidation.trajectoryInput = 'TrackRefitter'
process.TrackerOfflineValidation.moduleLevelHistsTransient = True
process.TrackerOfflineValidation.TH1ResModules = cms.PSet(
    xmin = cms.double(-0.5),
    Nbinx = cms.int32(300),
    xmax = cms.double(0.5)
)
process.TrackerOfflineValidation.TH1NormResModules = cms.PSet(
    xmin = cms.double(-3.0),
    Nbinx = cms.int32(300),
    xmax = cms.double(3.0)
)


##
## PATH
##
process.p = cms.Path(process.AlignmentTrackSelector*process.TrackRefitter*process.TrackerOfflineValidation)
