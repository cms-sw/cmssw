import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.Analyzer_cfi import TrackFindingTrackletAnalyzer_params
from L1Trigger.TrackFindingTracklet.Producer_cfi import TrackFindingTrackletProducer_params

TrackFindingTrackletAnalyzerTBout = cms.EDAnalyzer( 'trklet::AnalyzerTBout', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
TrackFindingTrackletAnalyzerTracklet = cms.EDAnalyzer( 'trklet::AnalyzerTracklet', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
TrackFindingTrackletAnalyzerKFin = cms.EDAnalyzer( 'trklet::AnalyzerKFin', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
TrackFindingTrackletAnalyzerKF = cms.EDAnalyzer( 'trackerTFP::AnalyzerKF', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
TrackFindingTrackletAnalyzerKFout = cms.EDAnalyzer( 'trklet::AnalyzerKFout', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
TrackFindingTrackletAnalyzerTT = cms.EDAnalyzer( 'trklet::AnalyzerTT', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
