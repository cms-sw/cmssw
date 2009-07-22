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
    '/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/468FB321-2060-DD11-8359-000423D6CAF2.root',
    '/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/5C2AE0AE-7D60-DD11-948C-001617DF785A.root',
    '/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/66CA0076-5060-DD11-8E3A-000423D99BF2.root',
    '/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/A00242A0-4F60-DD11-9D54-000423D98EC4.root',
    '/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/C4355E5F-1D60-DD11-82A7-000423D98B6C.root',
    '/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/FE4027D7-5060-DD11-87CE-000423D94E1C.root',
    '/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/26343D54-A760-DD11-9F64-001617E30CD4.root',
    '/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/9AA68E9A-B060-DD11-A61C-001617DBD556.root'
    )


)

##
## Maximum number of Events
##
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

##
## Output File Configuration
##
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('test_monitortrackresiduals.root')
)

##
## Load and Configure track selection for alignment
##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
#process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 1.3



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


# DQM services
process.load("DQMServices.Core.DQM_cfg")

# MonitorTrackResiduals
process.load("DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi")
process.MonitorTrackResiduals.OutputMEsInRootFile = True
process.MonitorTrackResiduals.Mod_On = False


process.p = cms.Path(process.AlignmentTrackSelector*process.TrackRefitter*process.MonitorTrackResiduals)
#process.MessageLogger.cerr.FwkReport.reportEvery = 10
#process.MessageLogger.cerr.threshold = 'Info'


