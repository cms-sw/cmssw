import FWCore.ParameterSet.Config as cms

l1EGammaEEProducer = cms.EDProducer("L1EGammaEEProducer",
                                    Multiclusters=cms.InputTag('hgcalTriggerPrimitiveDigiProducer:cluster3D'))
