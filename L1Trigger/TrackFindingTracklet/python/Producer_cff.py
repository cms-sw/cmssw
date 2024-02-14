import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.ProducerSetup_cff import TrackTriggerSetup
from L1Trigger.TrackerTFP.Producer_cfi import TrackerTFPProducer_params
from L1Trigger.TrackerTFP.ProducerES_cff import TrackTriggerDataFormats
from L1Trigger.TrackerTFP.ProducerLayerEncoding_cff import TrackTriggerLayerEncoding
from L1Trigger.TrackerTFP.KalmanFilterFormats_cff import TrackTriggerKalmanFilterFormats
from L1Trigger.TrackFindingTracklet.ChannelAssignment_cff import ChannelAssignment
from L1Trigger.TrackFindingTracklet.Producer_cfi import TrackFindingTrackletProducer_params

TrackFindingTrackletProducerIRin = cms.EDProducer( 'trklet::ProducerIRin', TrackFindingTrackletProducer_params )
TrackFindingTrackletProducerTBout = cms.EDProducer( 'trklet::ProducerTBout', TrackFindingTrackletProducer_params )
TrackFindingTrackletProducerDRin = cms.EDProducer( 'trklet::ProducerDRin', TrackFindingTrackletProducer_params )
TrackFindingTrackletProducerDR = cms.EDProducer( 'trklet::ProducerDR', TrackFindingTrackletProducer_params )
TrackFindingTrackletProducerKFin = cms.EDProducer( 'trklet::ProducerKFin', TrackFindingTrackletProducer_params )
TrackFindingTrackletProducerKF = cms.EDProducer( 'trackerTFP::ProducerKF', TrackFindingTrackletProducer_params )
TrackFindingTrackletProducerTT = cms.EDProducer( 'trklet::ProducerTT', TrackFindingTrackletProducer_params )
TrackFindingTrackletProducerAS = cms.EDProducer( 'trklet::ProducerAS', TrackFindingTrackletProducer_params )
TrackFindingTrackletProducerKFout = cms.EDProducer( 'trklet::ProducerKFout', TrackFindingTrackletProducer_params )