import FWCore.ParameterSet.Config as cms

dtChamberEfficiencyClient = cms.EDProducer("DTChamberEfficiencyClient",
                                           diagnosticPrescale = cms.untracked.int32(1))
