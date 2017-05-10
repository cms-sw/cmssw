import FWCore.ParameterSet.Config as cms

dataIntegrityTest = cms.EDProducer("DTDataIntegrityTest",
                                   diagnosticPrescale = cms.untracked.int32(1)
)


