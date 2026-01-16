# ESProducer providing the algorithm to run input data through modelsim and to compares results with expected output data
# and EDAnalyzer running the ESProduct produced by above ESProducer

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerTFP.Demonstrator_cfi import TrackTriggerDemonstrator_params
from L1Trigger.TrackerTFP.Producer_cfi import TrackerTFPProducer_params

TrackTriggerDemonstrator = cms.ESProducer("trackerTFP::ProducerDemonstrator", TrackTriggerDemonstrator_params)

TrackerTFPDemonstrator = cms.EDAnalyzer("trackerTFP::AnalyzerDemonstrator", TrackTriggerDemonstrator_params, TrackerTFPProducer_params)
