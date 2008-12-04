import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.fake.SiStripApvGainFakeESSource_cfi import *    
siStripApvGainFakeESSource.appendToDataLabel = 'fakeAPVGain'

from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
siStripGainESProducer.APVGain = 'fakeAPVGain'

import CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi
siStripGainESProducerforSimulation = CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi.siStripGainESProducer.clone()
es_prefer_siStripGainESProducer = cms.ESPrefer("SiStripGainESProducer","siStripGainESProducer")
siStripGainESProducerforSimulation.appendToDataLabel = 'fake'
siStripGainESProducerforSimulation.APVGain = 'fakeAPVGain'

