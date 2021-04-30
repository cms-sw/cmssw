import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1Taus.l1HPSPFTauProducer import l1HPSPFTauProducer as _l1HPSPFTauProducer
L1HPSPFTauProducerPF = _l1HPSPFTauProducer.clone(
  srcL1PFCands = "l1pfCandidates:PF"
)
