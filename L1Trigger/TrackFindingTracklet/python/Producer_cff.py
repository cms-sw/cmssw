# EDProducer to emulate hybrid track reconstruction chain after tracklet track fingding

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.Setup_cff import TrackFindingTrackletSetup
from L1Trigger.TrackFindingTracklet.DataFormats_cff import *
from L1Trigger.TrackFindingTracklet.KalmanFilterFormats_cfi import HybridKalmanFilterFormats_params
from L1Trigger.TrackFindingTracklet.Producer_cfi import TrackFindingTrackletProducer_params
from L1Trigger.TrackFindingTMTT.TMTrackProducer_Defaults_cfi import TMTrackProducer_params

ProducerIRin = cms.EDProducer( 'trklet::ProducerIRin', TrackFindingTrackletProducer_params )
ProducerTM = cms.EDProducer( 'trklet::ProducerTM', TrackFindingTrackletProducer_params )
ProducerDR = cms.EDProducer( 'trklet::ProducerDR', TrackFindingTrackletProducer_params )
ProducerKF = cms.EDProducer( 'trklet::ProducerKF', TrackFindingTrackletProducer_params, HybridKalmanFilterFormats_params, TMTrackProducer_params )
ProducerTQ = cms.EDProducer( 'trklet::ProducerTQ', TrackFindingTrackletProducer_params )
ProducerTFP = cms.EDProducer( 'trklet::ProducerTFP', TrackFindingTrackletProducer_params )

ProducerSim = cms.EDProducer( 'trklet::ProducerSim', TrackFindingTrackletProducer_params )
