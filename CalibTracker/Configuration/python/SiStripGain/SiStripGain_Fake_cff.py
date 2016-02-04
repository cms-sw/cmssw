import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.fake.SiStripApvGainFakeESSource_cfi import *    
siStripApvGainFakeESSource.appendToDataLabel = 'fakeAPVGain'

from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
siStripGainESProducer.APVGain = cms.PSet(
    Record = cms.string('fakeAPVGain'),
    Label = cms.untracked.string('')
)

import CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi
siStripGainESProducerforSimulation = CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi.siStripGainESProducer.clone()
es_prefer_siStripGainESProducer = cms.ESPrefer("SiStripGainESProducer","siStripGainESProducer")
siStripGainESProducerforSimulation.appendToDataLabel = 'fake'
siStripGainESProducerforSimulation.APVGain = siStripGainESProducer.APVGain = cms.PSet(
    Record = cms.string('fakeAPVGain'),
    Label = cms.untracked.string('')
)


