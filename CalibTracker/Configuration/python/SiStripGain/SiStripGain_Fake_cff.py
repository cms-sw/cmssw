import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.SiStripGainFakeSource_cfi import *
#siStripGainFakeESSource.appendToDataLabel = 'fakeAPV'

from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
siStripGainESProducer.APVGain = 'fakeAPVGain'

import CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi
siStripGainESProducerforSimulation = CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi.siStripGainESProducer.clone()
es_prefer_siStripGainESProducer = cms.ESPrefer("SiStripGainESProducer","siStripGainESProducer")
siStripGainESProducerforSimulation.appendToDataLabel = 'fake'
siStripGainESProducerforSimulation.APVGain = 'fakeAPVGain'

