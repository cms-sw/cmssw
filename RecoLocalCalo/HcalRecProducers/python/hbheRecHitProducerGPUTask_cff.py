import FWCore.ParameterSet.Config as cms

# Run 3 HCAL workflow on GPU

# EventSetup modules used by HBHERecHitProducerGPU
from RecoLocalCalo.HcalRecProducers.hcalGainsGPUESProducer_cfi import hcalGainsGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalGainWidthsGPUESProducer_cfi import hcalGainWidthsGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalLUTCorrsGPUESProducer_cfi import hcalLUTCorrsGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalConvertedPedestalsGPUESProducer_cfi import hcalConvertedPedestalsGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalConvertedEffectivePedestalsGPUESProducer_cfi import hcalConvertedEffectivePedestalsGPUESProducer
hcalConvertedEffectivePedestalsGPUESProducer.label0 = "withTopoEff"

from RecoLocalCalo.HcalRecProducers.hcalConvertedPedestalWidthsGPUESProducer_cfi import hcalConvertedPedestalWidthsGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalConvertedEffectivePedestalWidthsGPUESProducer_cfi import hcalConvertedEffectivePedestalWidthsGPUESProducer
hcalConvertedEffectivePedestalWidthsGPUESProducer.label0 = "withTopoEff"
hcalConvertedEffectivePedestalWidthsGPUESProducer.label1 = "withTopoEff"

from RecoLocalCalo.HcalRecProducers.hcalQIECodersGPUESProducer_cfi import hcalQIECodersGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalRecoParamsWithPulseShapesGPUESProducer_cfi import hcalRecoParamsWithPulseShapesGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalRespCorrsGPUESProducer_cfi import hcalRespCorrsGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalTimeCorrsGPUESProducer_cfi import hcalTimeCorrsGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalQIETypesGPUESProducer_cfi import hcalQIETypesGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalSiPMParametersGPUESProducer_cfi import hcalSiPMParametersGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalSiPMCharacteristicsGPUESProducer_cfi import hcalSiPMCharacteristicsGPUESProducer
from RecoLocalCalo.HcalRecProducers.hcalMahiPulseOffsetsGPUESProducer_cfi import hcalMahiPulseOffsetsGPUESProducer

# convert the HBHE digis into SoA format, and copy them from CPU to GPU
from EventFilter.HcalRawToDigi.hcalDigisProducerGPU_cfi import hcalDigisProducerGPU as _hcalDigisProducerGPU
hcalDigisGPU = _hcalDigisProducerGPU.clone(
    digisLabelF01HE = "",
    digisLabelF5HB = "",
    digisLabelF3HB = ""
)

# run the HCAL local reconstruction (MAHI) on GPU
from RecoLocalCalo.HcalRecProducers.hbheRecHitProducerGPU_cfi import hbheRecHitProducerGPU as _hbheRecHitProducerGPU
hbheRecHitProducerGPU = _hbheRecHitProducerGPU.clone(
    digisLabelF01HE = "hcalDigisGPU",
    digisLabelF5HB = "hcalDigisGPU",
    digisLabelF3HB = "hcalDigisGPU",
    recHitsLabelM0HBHE = ""
)

# Tasks and Sequences
hbheRecHitProducerGPUTask = cms.Task(
    hcalGainsGPUESProducer,
    hcalGainWidthsGPUESProducer,
    hcalLUTCorrsGPUESProducer,
    hcalConvertedPedestalsGPUESProducer,
    hcalConvertedEffectivePedestalsGPUESProducer,
    hcalConvertedPedestalWidthsGPUESProducer,
    hcalConvertedEffectivePedestalWidthsGPUESProducer,
    hcalQIECodersGPUESProducer,
    hcalRecoParamsWithPulseShapesGPUESProducer,
    hcalRespCorrsGPUESProducer,
    hcalTimeCorrsGPUESProducer,
    hcalQIETypesGPUESProducer,
    hcalSiPMParametersGPUESProducer,
    hcalSiPMCharacteristicsGPUESProducer,
    hcalMahiPulseOffsetsGPUESProducer,
    hcalDigisGPU,
    hbheRecHitProducerGPU
)

hbheRecHitProducerGPUSequence = cms.Sequence(hbheRecHitProducerGPUTask)
