import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1Taus.l1HPSPFTauProducer import l1HPSPFTauProducer as _l1HPSPFTauProducer
L1HPSPFTauProducerPuppi = _l1HPSPFTauProducer.clone(
  srcL1PFCands = "l1pfCandidates:Puppi",
  signalQualityCuts = dict(
    chargedHadron = dict(
      maxDz = 1.e+3
    ),
    muon = dict(
      maxDz = 1.e+3
    ),
    electron = dict(
      maxDz = 1.e+3
    )
  ),
  isolationQualityCuts = dict(
    chargedHadron = dict(
      maxDz = 1.e+3
    ),
    muon = dict(
      maxDz = 1.e+3
    ),
    electron = dict(
      maxDz = 1.e+3
    )
  )
)
