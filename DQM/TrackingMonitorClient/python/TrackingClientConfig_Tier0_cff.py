import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#  TrackingOfflineDQM (for Tier0 Harvesting Step) ####
trackingOfflineAnalyser = DQMEDHarvester("TrackingOfflineDQM",
    GlobalStatusFilling        = cms.untracked.int32(2),
    UsedWithEDMtoMEConverter   = cms.untracked.bool(True),
    TopFolderName              = cms.untracked.string("Tracking"),                                     
    TrackingGlobalQualityPSets = cms.VPSet(
         cms.PSet(
             QT         = cms.string("Rate"),
             dir        = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties"),
             name       = cms.string("NumberOfTracks_"),
         ),
         cms.PSet(
             QT         = cms.string("Chi2"),
             dir        = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties"),
             name       = cms.string("Chi2oNDF_"),
         ),
         cms.PSet(
             QT         = cms.string("RecHits"),
             dir        = cms.string("TrackParameters/highPurityTracks/pt_1/HitProperties"),
             name       = cms.string("NumberOfRecHitsPerTrack_"),
         ),
         cms.PSet(
             QT         = cms.string("Seed"),
             dir        = cms.string("TrackParameters/generalTracks/TrackBuilding"),
             name       = cms.string("NumberOfSeeds_"),
         )
    ),
    TrackingLSQualityPSets = cms.VPSet(
         cms.PSet(
             QT         = cms.string("Rate"),
             LSdir      = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties/LSanalysis"),
             LSname     = cms.string("NumberOfTracks_lumiFlag_"),
             LSlowerCut = cms.double(    1.0 ),
             LSupperCut = cms.double( 1000.0 )    
         ),
         cms.PSet(
             QT         = cms.string("Chi2"),
             LSdir      = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties/LSanalysis"),
             LSname     = cms.string("Chi2oNDF_lumiFlag_"),
             LSlowerCut = cms.double(  0.0 ),
             LSupperCut = cms.double( 25.0 )
         ),
         cms.PSet(
             QT         = cms.string("RecHits"),
             LSdir      = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties/LSanalysis"),
             LSname     = cms.string("NumberOfRecHitsPerTrack_lumiFlag_"),
             LSlowerCut = cms.double(  5.0 ),
             LSupperCut = cms.double( 20.0 )
         ),
         cms.PSet(
             QT         = cms.string("Seed"),
             LSdir      = cms.string("TrackParameters/generalTracks/LSanalysis"),
             LSname     = cms.string("NumberOfSeeds_lumiFlag_"),
             LSlowerCut = cms.double(       0.0 ),
             LSupperCut = cms.double( 1000000.0 )
         )
    )
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
trackingQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/TrackingMonitorClient/data/tracking_qualitytest_config_tier0.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    getQualityTestsFromFile = cms.untracked.bool(True)
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(trackingQTester,
                  qtList = cms.untracked.FileInPath('DQM/TrackingMonitorClient/data/tracking_qualitytest_config_tier0_heavyions.xml')
)

from DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfig_cff import trackingEffFromHitPattern
from DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfigZeroBias_cff import trackingEffFromHitPatternZeroBias

from DQM.TrackingMonitorClient.V0MonitoringClient_cff import *
from DQM.TrackingMonitorClient.primaryVertexResolutionClient_cfi import *
# Sequence

#import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
#TrackEffMon_ckf = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
#TrackEffMon_ckf.TKTrackCollection                  = 'ctfWithMaterialTracksP5'
#TrackEffMon_ckf.AlgoName                           = 'CKFTk'
#TrackEffMon_ckf.FolderName                         = 'Tracking/TrackParameters/TrackEfficiency'

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import *
TrackEffClient.FolderName = 'Tracking/TrackParameters/TrackEfficiency'
TrackEffClient.AlgoName   = 'CKFTk'

TrackingOfflineDQMClient = cms.Sequence(trackingQTester*trackingOfflineAnalyser*trackingEffFromHitPattern*voMonitoringClientSequence*primaryVertexResolutionClient*TrackEffClient)

TrackingOfflineDQMClientZeroBias = cms.Sequence(trackingQTester*trackingOfflineAnalyser*trackingEffFromHitPatternZeroBias*voMonitoringClientSequence*primaryVertexResolutionClient*TrackEffClient)
