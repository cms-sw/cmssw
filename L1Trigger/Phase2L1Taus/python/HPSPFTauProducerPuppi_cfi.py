import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1Taus.hpspfTauProducer_cfi import hpspfTauProducer as _hpspfTauProducer
HPSPFTauProducerPuppi = _hpspfTauProducer.clone(
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
