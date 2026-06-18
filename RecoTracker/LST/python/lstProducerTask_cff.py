import FWCore.ParameterSet.Config as cms

from RecoTracker.LST.lstProducer_cfi import lstProducer

from RecoTracker.LST.lstModulesDevESProducer_cfi import lstModulesDevESProducer

from RecoTracker.LST.lstInputProducer_cfi import lstInputProducer

from RecoTracker.LSTGeometry.lstGeometryESProducer_cfi import lstGeometryESProducer

lstProducerTask = cms.Task(lstGeometryESProducer, lstModulesDevESProducer, lstInputProducer, lstProducer)
