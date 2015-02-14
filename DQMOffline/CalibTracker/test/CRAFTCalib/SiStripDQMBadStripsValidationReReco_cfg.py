import FWCore.ParameterSet.Config as cms

process = cms.Process( "SiStripDQMBadStripsValidationReReco" )

### Miscellanous ###

## Logging ##

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool( True )
)
process.MessageLogger = cms.Service( "MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string( 'INFO' )
    )
)

## Profiling ##

# Memory #

process.SimpleMemoryCheck = cms.Service( "SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32( 0 )
)

### Import ###

## Magnetic fiels ##

process.load( "Configuration.StandardSequences.MagneticField_38T_cff" )

## Geometry ##

process.load( "Configuration.StandardSequences.GeometryRecoDB_cff" )

## Calibration ##

# Global tag #
 
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
process.GlobalTag.globaltag = 'CRAFT_ALL_V4::All'
process.es_prefer_GlobalTag = cms.ESPrefer( 'PoolDBESSource', 'GlobalTag' )

### SiStrip DQM ###

## Reconstruction ##

process.load( "RecoTracker.Configuration.RecoTrackerP5_cff" )

## DQM modules ##

# SiStripMonitorCluster #

import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
process.siStripMonitorCluster = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
process.siStripMonitorCluster.OutputMEsInRootFile                     = False 
process.siStripMonitorCluster.SelectAllDetectors                      = True 
process.siStripMonitorCluster.StripQualityLabel                       = ''   
process.siStripMonitorCluster.TH1ClusterPos.moduleswitchon            = True 
process.siStripMonitorCluster.TH1nClusters.layerswitchon              = True 
process.siStripMonitorCluster.TH1nClusters.moduleswitchon             = False
process.siStripMonitorCluster.TH1ClusterStoN.moduleswitchon           = False
process.siStripMonitorCluster.TH1ClusterStoNVsPos.moduleswitchon      = True 
process.siStripMonitorCluster.TH1ClusterNoise.moduleswitchon          = False
process.siStripMonitorCluster.TH1NrOfClusterizedStrips.moduleswitchon = False
process.siStripMonitorCluster.TH1ModuleLocalOccupancy.moduleswitchon  = False
process.siStripMonitorCluster.TH1ClusterCharge.moduleswitchon         = False
process.siStripMonitorCluster.TH1ClusterWidth.moduleswitchon          = False

# SiStripMonitorTrack #

import RecoTracker.TrackProducer.RefitterWithMaterial_cfi
process.ctfWithMaterialTracksP5Refitter                   = RecoTracker.TrackProducer.RefitterWithMaterial_cfi.TrackRefitter.clone()
process.ctfWithMaterialTracksP5Refitter.src               = 'ctfWithMaterialTracksP5'
process.ctfWithMaterialTracksP5Refitter.TrajectoryInEvent = True
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
process.SiStripMonitorTrackReal = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
process.SiStripMonitorTrackReal.TrackProducer = 'ctfWithMaterialTracksP5'
process.SiStripMonitorTrackReal.TrackLabel    = ''
process.SiStripMonitorTrackReal.Cluster_src   = 'siStripClusters'
process.SiStripMonitorTrackReal.FolderName    = 'SiStrip/Tracks'
# process.SiStripMonitorTrackReal.Mod_On        = True

### Input ###

## PoolSource ## 

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/DE20B094-1FC2-DD11-90AC-001D0967D5A8.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0004/6C368AB3-96C2-DD11-BDAC-001D0967CF86.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/1A572E65-75C4-DD11-8E97-001D0967C64E.root',
        '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0010/52664B9C-E8C4-DD11-B292-0019B9E48877.root'
    ),
    skipEvents = cms.untracked.uint32(0)
)

## Input steering ##

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 1000 )
)

### Output ###

## DQM ##

process.load( "DQMServices.Core.DQM_cfg" )
process.DQM.collectorHost = ''
process.load( "DQMServices.Components.DQMEnvironment_cfi" )
process.dqmSaver.convention        = 'Online'
process.dqmSaver.dirName           = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRACKER/DQM/SiStrip/jobs/output'
process.dqmSaver.producer          = 'DQM'
process.dqmSaver.saveByRun         = 1
process.dqmSaver.saveAtJobEnd      = True
process.dqmSaver.referenceHandling = 'qtests'
process.dqmEnv.subSystemFolder = 'SiStrip'

### Scheduling ###

## Paths ##

# DQM path #

process.p = cms.Path(
    process.siStripMonitorCluster  *
    process.ctfWithMaterialTracksP5Refitter *
    process.SiStripMonitorTrackReal *
    process.dqmSaver
)
