import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
hcalGlobalRecoTask = cms.Task(hbhereco)
hcalGlobalRecoSequence = cms.Sequence(hcalGlobalRecoTask)

from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toReplaceWith( hbhereco, _phase1_hbheprereco ) # >=Run3
