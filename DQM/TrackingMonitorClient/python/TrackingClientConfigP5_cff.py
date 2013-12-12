import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSummary.OnDemandMonitoring_cfi import *
#  TrackingMonitorAnalyser ####
TrackingAnalyser = cms.EDAnalyzer("TrackingAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(1),
    GlobalStatusFilling      = cms.untracked.int32(2),
    TkMapCreationFrequency   = cms.untracked.int32(3),
    SummaryCreationFrequency = cms.untracked.int32(5),
    ShiftReportFrequency     = cms.untracked.int32(-1),
    SummaryConfigPath        = cms.untracked.string("DQM/TrackingMonitorClient/data/tracking_monitorelement_config.xml"),
    PrintFaultyModuleList    = cms.untracked.bool(True),                                
    RawDataTag               = cms.untracked.InputTag("source"),                              
    TrackRatePSet            = cms.PSet(
           Name     = cms.string("NumberOfGoodTracks_"),
                  LowerCut = cms.double(1.0),
                  UpperCut = cms.double(1000.0),
               ),
                                            TrackChi2PSet            = cms.PSet(
           Name     = cms.string("GoodTrackChi2oNDF_"),
                  LowerCut = cms.double(0.0),
                  UpperCut = cms.double(25.0),
               ),
                                            TrackHitPSet            = cms.PSet(
           Name     = cms.string("GoodTrackNumberOfRecHitsPerTrack_"),
                  LowerCut = cms.double(5.0),
                  UpperCut = cms.double(20.0),
               ),
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        loadFecCabling = cms.untracked.bool(True),
        loadLVCabling  = cms.untracked.bool(True),
        loadHVCabling  = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/Integration/test/TkMap/')
    )
)
# Track Efficiency Client

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import *
TrackEffClient.FolderName = 'Tracking/TrackParameters/TrackEfficiency'
TrackEffClient.AlgoName   = 'CKFTk'

