import FWCore.ParameterSet.Config as cms

#  TrackingMonitorAnalyser ####
TrackingAnalyserCosmic = cms.EDAnalyzer("TrackingAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(1),
    GlobalStatusFilling      = cms.untracked.int32(2),
    ShiftReportFrequency     = cms.untracked.int32(-1),
    RawDataTag               = cms.untracked.InputTag("source"),                              
    TopFolderName              = cms.untracked.string("Tracking"),
    TrackingGlobalQualityPSets = cms.VPSet(
         cms.PSet(
             QT         = cms.string("Rate"),
             dir        = cms.string("TrackParameters/GeneralProperties"),
             name       = cms.string("NumberOfTracks_"),
         ),
         cms.PSet(
             QT         = cms.string("Chi2"),
             dir        = cms.string("TrackParameters/GeneralProperties"),
             name       = cms.string("TrackChi2oNDF_"),
         ),
         cms.PSet(
             QT         = cms.string("RecHits"),
             dir        = cms.string("TrackParameters/HitProperties"),
             name       = cms.string("TrackNumberOfRecHitsPerTrack_"),
         ),
    ),
    TrackingLSQualityPSets = cms.VPSet(
         cms.PSet(
             QT         = cms.string("Rate"),
             LSdir      = cms.string("TrackParameters/GeneralProperties/LSanalysis"),
             LSname     = cms.string("NumberOfTracks_lumiFlag_"),
             LSlowerCut = cms.double( -1.0 ),
             LSupperCut = cms.double(  1.0 )    
         ),
         cms.PSet(
             QT         = cms.string("Chi2"),
             LSdir      = cms.string("TrackParameters/GeneralProperties/LSanalysis"),
             LSname     = cms.string("TrackChi2oNDF_lumiFlag_"),
             LSlowerCut = cms.double(  0.0 ),
             LSupperCut = cms.double( 25.0 )
         ),
         cms.PSet(
             QT         = cms.string("RecHits"),
             LSdir      = cms.string("TrackParameters/GeneralProperties/LSanalysis"),
             LSname     = cms.string("TrackNumberOfRecHitsPerTrack_lumiFlag_"),
             LSlowerCut = cms.double(  3.0 ),
             LSupperCut = cms.double( 35.0 )
         ),
    )
)

# Track Efficiency Client

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import *
TrackEffClient.FolderName = 'Tracking/TrackParameters/TrackEfficiency'
TrackEffClient.AlgoName   = 'CKFTk'
