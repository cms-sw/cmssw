import FWCore.ParameterSet.Config as cms

from RecoTracker.LST.lstProducer_cfi import lstProducer

from RecoTracker.LST.lstModulesDevESProducer_cfi import lstModulesDevESProducer

# not scheduled task to get the framework to hide the producer
dummyLSTModulesDevESProducerTask = cms.Task(lstModulesDevESProducer)
