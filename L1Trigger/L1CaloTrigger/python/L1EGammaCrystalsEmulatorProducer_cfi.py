import FWCore.ParameterSet.Config as cms

L1EGammaClusterEmuProducer = cms.EDProducer("L1EGCrystalClusterEmulatorProducer",
   ecalTPEB = cms.InputTag("simEcalEBTriggerPrimitiveDigis","","HLT"),
   hcalTP = cms.InputTag("simHcalTriggerPrimitiveDigis","","HLT"),
)
