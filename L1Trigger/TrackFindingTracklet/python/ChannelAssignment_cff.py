# ESProducer providing the algorithm to assign tracklet tracks and stubs to output channel based on their Pt or seed type as well as DTC stubs to input channel

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackFindingTracklet.ChannelAssignment_cfi import ChannelAssignment_params

ChannelAssignment = cms.ESProducer("trklet::ProducerChannelAssignment", ChannelAssignment_params)
