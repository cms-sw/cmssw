import FWCore.ParameterSet.Config as cms

# RecHits options
siPhase2RecHits = cms.EDProducer("Phase2TrackerRecHits",
  src = cms.InputTag("siPhase2Clusters"),
  Phase2StripCPE = cms.ESInputTag("phase2StripCPEESProducer", "Phase2StripCPETrivial")
)
