import FWCore.ParameterSet.Config as cms

slimmedMETs = cms.EDProducer("PATMETSlimmer",
   src = cms.InputTag("patMETs"),
   rawUncertainties   = cms.InputTag("patPFMet%s"),
   type1Uncertainties = cms.InputTag("patType1CorrectedPFMet%s"),
   type1p2Uncertainties = cms.InputTag("patType1p2CorrectedPFMet%s"),
)

