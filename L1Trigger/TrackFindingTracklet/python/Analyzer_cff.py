import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.Analyzer_cfi import TrackFindingTrackletAnalyzer_params
from L1Trigger.TrackFindingTracklet.ProducerKF_cfi import TrackFindingTrackletProducerKF_params

TrackFindingTrackletAnalyzerTracklet = cms.EDAnalyzer( 'trklet::AnalyzerTracklet', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducerKF_params )
TrackFindingTrackletAnalyzerKFin = cms.EDAnalyzer( 'trklet::AnalyzerKFin', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducerKF_params )
TrackFindingTrackletAnalyzerKF = cms.EDAnalyzer( 'trackerTFP::AnalyzerKF', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducerKF_params )
TrackFindingTrackletAnalyzerKFout = cms.EDAnalyzer( 'trklet::AnalyzerKFout', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducerKF_params )
TrackFindingTrackletAnalyzerTT = cms.EDAnalyzer( 'trklet::AnalyzerTT', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducerKF_params )