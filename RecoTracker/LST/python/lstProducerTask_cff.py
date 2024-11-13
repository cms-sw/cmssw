import FWCore.ParameterSet.Config as cms

from RecoTracker.LST.lstProducer_cfi import lstProducer

from RecoTracker.LST.lstModulesDevESProducer_cfi import lstModulesDevESProducer

lstProducerTask = cms.Task(lstModulesDevESProducer, lstProducer)
