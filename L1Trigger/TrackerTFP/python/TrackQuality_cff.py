# ESProducer providing Bit accurate emulation of the track quality BDT

import FWCore.ParameterSet.Config as cms
from L1Trigger.TrackerTFP.TrackQuality_cfi import TrackQuality_params

TrackTriggerTrackQuality = cms.ESProducer("trackerTFP::ProducerTrackQuality", TrackQuality_params)

