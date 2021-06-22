import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#from DQM.TrackingMonitorSummary.OnDemandMonitoring_cfi import *
#  TrackingMonitorAnalyser ####
TrackingAnalyser = DQMEDHarvester("TrackingAnalyser",
    nFEDinfoDir              = cms.string("SiStrip/FEDIntegrity_SM"),                                   
    nFEDinVsLSname           = cms.string("nFEDinVsLS"),
    nFEDinWdataVsLSname      = cms.string("nFEDinWdataVsLS"),
    StaticUpdateFrequency    = cms.untracked.int32(1),
    GlobalStatusFilling      = cms.untracked.int32(2),
    ShiftReportFrequency     = cms.untracked.int32(-1),
    RawDataTag               = cms.untracked.InputTag("source"),                              
    TopFolderName              = cms.untracked.string("Tracking"),
    verbose                  = cms.untracked.bool(False),
    TrackingGlobalQualityPSets = cms.VPSet(
         cms.PSet(
             QT         = cms.string("Rate"),
             dir        = cms.string("TrackParameters/GeneralProperties/"),
             name       = cms.string("NumberOfTracks_"),
         ),
         cms.PSet(
             QT         = cms.string("Chi2"),
             dir        = cms.string("TrackParameters/GeneralProperties/"),
             name       = cms.string("Chi2oNDF_"),
         ),
         cms.PSet(
             QT         = cms.string("RecHits"),
             dir        = cms.string("TrackParameters/HitProperties/"),
             name       = cms.string("NumberOfRecHitsPerTrack_"),
         ),
    ),
    TrackingLSQualityPSets = cms.VPSet(
         cms.PSet(
             QT         = cms.string("Rate"),
             LSdir      = cms.string("TrackParameters/GeneralProperties/LSanalysis"),
             LSname     = cms.string("NumberOfTracks_lumiFlag_"),
             LSlowerCut = cms.double(    1.0 ),
             LSupperCut = cms.double( 1000.0 )    
         ),
         cms.PSet(
             QT         = cms.string("Chi2"),
             LSdir      = cms.string("TrackParameters/GeneralProperties/LSanalysis"),
             LSname     = cms.string("Chi2oNDF_lumiFlag_"),
             LSlowerCut = cms.double(  0.0 ),
             LSupperCut = cms.double( 25.0 )
         ),
         cms.PSet(
             QT         = cms.string("RecHits"),
             LSdir      = cms.string("TrackParameters/GeneralProperties/LSanalysis"),
             LSname     = cms.string("NumberOfRecHitsPerTrack_lumiFlag_"),
             LSlowerCut = cms.double(  5.0 ),
             LSupperCut = cms.double( 20.0 )
         ),
    )
)

# Track Efficiency Client

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import *
TrackEffClient.FolderName = 'Tracking/TrackParameters/TrackEfficiency'
TrackEffClient.AlgoName   = 'CKFTk'

