import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi

TrackMonSeed = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()

TrackMonSeed.OutputMEsInRootFile        = cms.bool(False)
TrackMonSeed.OutputFileName             = cms.string('TrackingMonitorSeedMultiplicity.root')
TrackMonSeed.MeasurementState           = cms.string('ImpactPoint')
TrackMonSeed.FolderName                 = cms.string('Tracking/TrackParameters')
TrackMonSeed.BSFolderName               = cms.string('Tracking/TrackParameters/BeamSpotParameters')
TrackMonSeed.AlgoName                   = cms.string('Seed')
#TrackMonSeed.doGoodTrackPlots       = cms.bool(False)
TrackMonSeed.doTrackerSpecific          = cms.bool(False)
TrackMonSeed.doAllPlots                 = cms.bool(False)
TrackMonSeed.doHitPropertiesPlots       = cms.bool(False)
TrackMonSeed.doGeneralPropertiesPlots   = cms.bool(False)
TrackMonSeed.doBeamSpotPlots            = cms.bool(False)
TrackMonSeed.doSeedParameterHistos      = cms.bool(False)
TrackMonSeed.doLumiAnalysis             = cms.bool(False)
TrackMonSeed.doMeasurementStatePlots    = cms.bool(False)
TrackMonSeed.doRecHitsPerTrackProfile   = cms.bool(False)
TrackMonSeed.doRecHitVsPhiVsEtaPerTrack = cms.bool(False)
#TrackMonSeed.doGoodTrackRecHitVsPhiVsEtaPerTrack = cms.bool(False)
#
# plot on Seed (total number, pt, seed # vs cluster)
#
TrackMonSeed.doSeedNumberHisto    = cms.bool(True)
TrackMonSeed.doSeedLumiAnalysis   = cms.bool(True)
TrackMonSeed.doSeedVsClusterHisto = cms.bool(True)
TrackMonSeed.doSeedPTHisto        = cms.bool(True)
TrackMonSeed.doSeedETAHisto       = cms.bool(True)
TrackMonSeed.doSeedPHIHisto       = cms.bool(True)
TrackMonSeed.doSeedPHIVsETAHisto  = cms.bool(True)
TrackMonSeed.doStopSource         = cms.bool(True)
