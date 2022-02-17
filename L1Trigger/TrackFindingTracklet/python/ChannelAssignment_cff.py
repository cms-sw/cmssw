import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.ChannelAssignment_cfi import ChannelAssignment_params

ChannelAssignment = cms.ESProducer("trklet::ProducerChannelAssignment", ChannelAssignment_params)
