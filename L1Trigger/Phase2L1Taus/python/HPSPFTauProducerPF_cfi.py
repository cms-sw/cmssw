import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1Taus.hpspfTauProducer_cfi import hpspfTauProducer as _hpspfTauProducer
l1tHPSPFTauProducerPF = _hpspfTauProducer.clone(
    srcL1PFCands = "l1tLayer1:PF",
)
