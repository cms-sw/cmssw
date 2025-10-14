import FWCore.ParameterSet.Config as cms

#--- for Run 1 and Run 2
from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import hbhereco as _phase0_hbhereco
hbhereco = _phase0_hbhereco.clone()
hbherecoLegacy = _phase0_hbhereco.clone()

hcalGlobalRecoTask = cms.Task(hbhereco)
hcalGlobalRecoSequence = cms.Sequence(hcalGlobalRecoTask)

hcalOnlyGlobalRecoTask = cms.Task()
hcalOnlyGlobalRecoSequence = cms.Sequence(hcalOnlyGlobalRecoTask)

#-- Legacy HCAL Only Task
hcalOnlyLegacyGlobalRecoTask = cms.Task() 

#--- for Run 3 and later
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB

from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco
run3_HB.toReplaceWith(hbhereco, _phase1_hbheprereco)
run3_HB.toReplaceWith(hcalOnlyGlobalRecoTask, cms.Task(hbhereco))
run3_HB.toReplaceWith(hbherecoLegacy, _phase1_hbheprereco)
run3_HB.toReplaceWith(hcalOnlyLegacyGlobalRecoTask, cms.Task(hbherecoLegacy))

#--- for Run 3 on GPU
from Configuration.ProcessModifiers.alpaka_cff import alpaka

from RecoLocalCalo.HcalRecProducers.hcalRecHitSoAToLegacy_cfi import  hcalRecHitSoAToLegacy 
(alpaka & run3_HB).toReplaceWith(hbhereco,
    hcalRecHitSoAToLegacy.clone(
        src = ("hbheRecHitProducerPortable","")
    )
)

hbherecoSerial = hcalRecHitSoAToLegacy.clone(
    src = ("hbheRecHitProducerSerial","")
)
alpaka.toReplaceWith(hcalGlobalRecoTask, hcalGlobalRecoTask.copyAndAdd(hbherecoSerial))
alpaka.toReplaceWith(hcalOnlyGlobalRecoTask, hcalOnlyGlobalRecoTask.copyAndAdd(hbherecoSerial))

##
## Modify for the tau embedding methods cleaning step
##
from Configuration.ProcessModifiers.tau_embedding_cleaning_cff import tau_embedding_cleaning
from TauAnalysis.MCEmbeddingTools.Cleaning_RECO_cff import tau_embedding_hbhereco_cleaner
tau_embedding_cleaning.toReplaceWith(hbhereco, tau_embedding_hbhereco_cleaner)