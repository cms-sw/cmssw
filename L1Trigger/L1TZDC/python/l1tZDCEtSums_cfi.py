import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TZDC.l1tZDCEtSumsProducer_cfi import l1tZDCEtSumsProducer as _l1tZDCEtSumsProducer

l1tZDCEtSums = _l1tZDCEtSumsProducer.clone(
    bxLast = 3
)
