# ESProducer providing layer id encoding (Layers consitent with rough r-z track parameters are counted from 0 onwards) used by Kalman Filter

import FWCore.ParameterSet.Config as cms

TrackTriggerLayerEncoding = cms.ESProducer("trackerTFP::ProducerLayerEncoding")
