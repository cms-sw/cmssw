import FWCore.ParameterSet.Config as cms

dtChamberEfficiencyClient = cms.EDAnalyzer("DTChamberEfficiencyClient",
                                           diagnosticPrescale = cms.untracked.int32(1))
