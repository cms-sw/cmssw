import FWCore.ParameterSet.Config as cms

# RecHits options
hltSiPhase2RecHits = cms.EDProducer("Phase2TrackerRecHits",
  src = cms.InputTag("hltSiPhase2Clusters"),
  Phase2StripCPE = cms.ESInputTag("hltESPPhase2StripCPE", "Phase2StripCPEHLT")
)
