import FWCore.ParameterSet.Config as cms

slimmedMETs = cms.EDProducer("PATMETSlimmer",
   src = cms.InputTag("patMETs"),
   rawUncertainties   = cms.InputTag("patPFMet%s"),
   type1Uncertainties = cms.InputTag("patPFMetT1%s"),
   type1p2Uncertainties = cms.InputTag("patPFMetT1T2%s"),

   #caloMET, will be used for the beginning of ata takin by the JetMET people
   caloMET = cms.InputTag("patCaloMet"),
)

