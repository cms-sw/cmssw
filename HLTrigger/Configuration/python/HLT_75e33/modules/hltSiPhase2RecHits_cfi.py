import FWCore.ParameterSet.Config as cms

# RecHits options
hltSiPhase2RecHits = cms.EDProducer("Phase2TrackerRecHits",
  src = cms.InputTag("hltSiPhase2Clusters"),
  Phase2StripCPE = cms.ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE")
)
