# EDAnalyzer to analyze hybrid track reconstruction emulation chain

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.Analyzer_cfi import TrackFindingTrackletAnalyzer_params
from L1Trigger.TrackFindingTracklet.Producer_cfi import TrackFindingTrackletProducer_params
from L1Trigger.TrackFindingTracklet.ChannelAssignment_cfi import ChannelAssignment_params
from SimTracker.TrackTriggerAssociation.StubAssociator_cfi import StubAssociator_params

AnalyzerTQ = cms.EDAnalyzer( 'trklet::AnalyzerTQ', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params, StubAssociator_params )
AnalyzerTB = cms.EDAnalyzer( 'trklet::AnalyzerTB', TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params, StubAssociator_params )

AnalyzerStream  = cms.EDAnalyzer( 'tt::AnalyzerStreamTrack',  TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params, StubAssociator_params )
AnalyzerTTTrack = cms.EDAnalyzer( 'tt::AnalyzerTTTrack',      TrackFindingTrackletAnalyzer_params, TrackFindingTrackletProducer_params, StubAssociator_params )

AnalyzerTM = AnalyzerStream.clone( Process = "TM", NumLayers = ChannelAssignment_params.TM.NumLayers )
AnalyzerDR = AnalyzerStream.clone( Process = "DR" )
AnalyzerKF = AnalyzerStream.clone( Process = "KF" )

AnalyzerTracklet = AnalyzerTTTrack.clone( InputTag = cms.InputTag( "l1tTTTracksFromTrackletEmulation", "Level1TTTracks"  ), Process = "Tracklet" )
AnalyzerTFP      = AnalyzerTTTrack.clone( InputTag = cms.InputTag( "ProducerTFP",                      "TTTrackAccepted" ), Process = "TFP"      )
