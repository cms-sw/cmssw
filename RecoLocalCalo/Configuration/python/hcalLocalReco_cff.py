import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
hcalOOTPileupESProducer = cms.ESProducer('OOTPileupDBCompatibilityESProducer')

from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi import *

hcalLocalRecoSequence = cms.Sequence(hbheprereco+hfreco+horeco+zdcreco)

from Configuration.StandardSequences.Eras import eras

def _modifyHcalLocalRecoForHF2016( theProcess ):
    theProcess.load("RecoLocalCalo.HcalRecProducers.HFQIE10Reconstructor_cfi")
    theProcess.hfreco = theProcess.hfQIE10Reco.clone()
    hcalLocalRecoSequence.replace(hfreco,theProcess.hfreco)

modifyRecoLocalCaloConfigurationHcalLocalRecoForHF2016 = eras.run2_HF_2016.makeProcessModifier( _modifyHcalLocalRecoForHF2016 )

#from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
#hcalGlobalRecoSequence = cms.Sequence(hbhereco)
