import FWCore.ParameterSet.Config as cms

# Run 3 HCAL workflow on GPU

# EventSetup modules used by HBHERecHitProducerPortable

from RecoLocalCalo.HcalRecProducers.hcalMahiConditionsESProducer_cfi import hcalMahiConditionsESProducer
from RecoLocalCalo.HcalRecProducers.hcalSiPMCharacteristicsESProducer_cfi import hcalSiPMCharacteristicsESProducer 
from RecoLocalCalo.HcalRecAlgos.hcalRecoParamWithPulseShapeESProducer_cfi import hcalRecoParamWithPulseShapeESProducer 

# convert the HBHE digis into SoA format
from EventFilter.HcalRawToDigi.hcalDigisSoAProducer_cfi import hcalDigisSoAProducer as _hcalDigisSoAProducer
hcalDigisPortable = _hcalDigisSoAProducer.clone(
    digisLabelF01HE = "f01HEDigis",
    digisLabelF5HB  = "f5HBDigis",
    digisLabelF3HB  = "f3HBDigis"
)

from HeterogeneousCore.AlpakaCore.functions import *
hcalDigisSerial = makeSerialClone(hcalDigisPortable)

# run the HCAL local reconstruction (MAHI) on GPU
from RecoLocalCalo.HcalRecProducers.hbheRecHitProducerPortable_cfi import hbheRecHitProducerPortable as _hbheRecHitProducerPortable
hbheRecHitProducerPortable = _hbheRecHitProducerPortable.clone(
    digisLabelF01HE = ("hcalDigisPortable", "f01HEDigis"),
    digisLabelF5HB = ("hcalDigisPortable", "f5HBDigis"),
    digisLabelF3HB = ("hcalDigisPortable","f3HBDigis"),
    recHitsLabelM0HBHE = "",
)
hbheRecHitProducerSerial = makeSerialClone(hbheRecHitProducerPortable,
    digisLabelF01HE = ("hcalDigisSerial","f01HEDigis"),
    digisLabelF5HB = ("hcalDigisSerial","f5HBDigis"),
    digisLabelF3HB = ("hcalDigisSerial","f3HBDigis")
)

# Tasks and Sequences
hbheRecHitProducerPortableTask = cms.Task(
    hcalMahiConditionsESProducer,
    hcalRecoParamWithPulseShapeESProducer,
    hcalSiPMCharacteristicsESProducer,
    hcalDigisPortable,
    hcalDigisSerial,
    hbheRecHitProducerPortable,
    hbheRecHitProducerSerial
)

hbheRecHitProducerPortableSequence = cms.Sequence(hbheRecHitProducerPortableTask)
