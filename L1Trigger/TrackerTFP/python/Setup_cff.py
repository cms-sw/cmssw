# ESProducer processing and providing run-time constants used by Track Trigger emulators

import FWCore.ParameterSet.Config as cms
from L1Trigger.TrackerTFP.Setup_cfi import TrackerTFP_params

TrackerTFPSetup = cms.ESProducer("trackerTFP::ProducerSetup", TrackerTFP_params)
