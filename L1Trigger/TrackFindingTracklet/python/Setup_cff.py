# ESProducer processing and providing run-time constants used by Track Trigger emulators

import FWCore.ParameterSet.Config as cms
from L1Trigger.TrackFindingTracklet.Setup_cfi import TrackFindingTracklet_params

TrackFindingTrackletSetup = cms.ESProducer("trklet::ProducerSetup", TrackFindingTracklet_params)
