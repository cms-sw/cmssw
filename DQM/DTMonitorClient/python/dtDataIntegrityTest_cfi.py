import FWCore.ParameterSet.Config as cms

dataIntegrityTest = cms.EDAnalyzer("DTDataIntegrityTest",
                                   diagnosticPrescale = cms.untracked.int32(1)
)


