import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TZDC.l1tZDCProducer_cfi import l1tZDCProducer as _l1tZDCProducer

l1tZDCEtSums = _l1tZDCProducer.clone(
    bxLast = 3
)
