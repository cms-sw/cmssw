import FWCore.ParameterSet.Config as cms

particleFlowTmpPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
                                     src = cms.InputTag('particleFlowTmp')
)



