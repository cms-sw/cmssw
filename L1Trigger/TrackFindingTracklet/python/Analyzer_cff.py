# EDAnalyzer to analyze hybrid track reconstruction emulation chain

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.Analyzer_cfi import TrackFindingTrackletAnalyzer_params
from L1Trigger.TrackFindingTracklet.Producer_cfi import TrackFindingTrackletProducer_params
from L1Trigger.TrackFindingTracklet.ChannelAssignment_cff import *
from L1Trigger.TrackFindingTracklet.DataFormats_cff import *

AnalyzerTracklet = cms.EDAnalyzer( 'trklet::AnalyzerTracklet', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
AnalyzerTM  = cms.EDAnalyzer( 'trklet::AnalyzerTM',  TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
AnalyzerDR  = cms.EDAnalyzer( 'trklet::AnalyzerDR',  TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
AnalyzerKF  = cms.EDAnalyzer( 'trklet::AnalyzerKF',  TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
AnalyzerTQ  = cms.EDAnalyzer( 'trackerTFP::AnalyzerTQ',  TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
AnalyzerTFP = cms.EDAnalyzer( 'trackerTFP::AnalyzerTFP', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params )
