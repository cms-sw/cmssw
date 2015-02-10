import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")


##
## Message Logger
##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
# report only every 100th record
process.MessageLogger.cerr.FwkReport.reportEvery = 100


## 
## Database configuration
##
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.load("CondCore.DBCommon.CondDBSetup_cfi")


##
## Fake Conditions (if needed)
##
# For older CMSSW versions
#process.load("Configuration.StandardSequences.FakeConditions_cff")
# Compatible to CMSSW_3_1_1
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'DESIGN_31X_V2::All'


##
## Geometry
##
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")


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
    ## 311 RelVal Samples (9000 events)
    ##
    '/store/relval/CMSSW_3_1_1/RelValMinBias/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/B4AF1681-D66B-DE11-B20A-001D09F2924F.root',
    '/store/relval/CMSSW_3_1_1/RelValMinBias/GEN-SIM-RECO/STARTUP31X_V1-v2/0002/0A9066D1-B96B-DE11-91A6-000423D99160.root',
    ##
    ## 311 RelVal Samples (cosmics: need to set trackCollection to eg. process.AlignmentTrackSelector.src = "ctfWithMaterialTracksP5")
    ##
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/FED903D5-3E6C-DE11-8C44-001D09F2432B.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/FCC739FF-426C-DE11-B074-001D09F2545B.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/F21CB2E5-486C-DE11-AB26-000423D6B5C4.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/F0EAF77B-416C-DE11-94BE-001D09F2545B.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/F0C3CF4C-426C-DE11-A114-0030487A3232.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/EEDA4BFB-4D6C-DE11-AB6D-000423D6CA02.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/EE789677-3E6C-DE11-9767-001D09F295A1.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/ECC8A34E-486C-DE11-87F4-000423D6CA72.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/EC091E8E-436C-DE11-985D-001D09F2545B.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/E64146C5-436C-DE11-83BE-0030487A1990.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/E4F9807A-486C-DE11-8A94-001D09F2514F.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/E4BCCE62-4A6C-DE11-BDBD-000423D6CA42.root',
    #'/store/relval/CMSSW_3_1_1/RelValCosmics/RECO/CRAFT0831X_V1-v1/0003/E2E2FC2B-436C-DE11-8C4A-001D09F2545B.root',
    ##
    ## 21X RelVal Samples
    ##
    #'/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/468FB321-2060-DD11-8359-000423D6CAF2.root',
    #'/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/5C2AE0AE-7D60-DD11-948C-001617DF785A.root',
    #'/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/66CA0076-5060-DD11-8E3A-000423D99BF2.root',
    #'/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/A00242A0-4F60-DD11-9D54-000423D98EC4.root',
    #'/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/C4355E5F-1D60-DD11-82A7-000423D98B6C.root',
    #'/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/FE4027D7-5060-DD11-87CE-000423D94E1C.root',
    #'/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/26343D54-A760-DD11-9F64-001617E30CD4.root',
    #'/store/relval/CMSSW_2_1_0/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0001/9AA68E9A-B060-DD11-A61C-001617DBD556.root'
    )
)


##
## Maximum number of Events
##
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)


##
## Load and Configure track selection for alignment
##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
#process.AlignmentTrackSelector.src = "ctfWithMaterialTracksP5" ## to set for cosmics
#process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 1.3


##
## Load and Configure TrackRefitter
##
# For older CMSSW versions
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
#process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")
# Compatible to CMSSW_3_1_1
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True


##
## DQM services
##
process.load("DQMServices.Core.DQM_cfg")


##
## MonitorTrackResiduals
##
process.load("DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi")
process.MonitorTrackResiduals.OutputMEsInRootFile = True
process.MonitorTrackResiduals.Mod_On = False


##
## Sequence
##
process.p = cms.Path(process.AlignmentTrackSelector*process.TrackRefitter*process.MonitorTrackResiduals)
#process.MessageLogger.cerr.FwkReport.reportEvery = 10
#process.MessageLogger.cerr.threshold = 'Info'


