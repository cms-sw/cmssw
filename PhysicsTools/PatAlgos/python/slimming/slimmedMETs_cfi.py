import FWCore.ParameterSet.Config as cms

slimmedMETs = cms.EDProducer("PATMETSlimmer",
   src = cms.InputTag("patMETs"),                
   rawVariation   = cms.InputTag("patPFMet"),
   t1Uncertainties  = cms.InputTag("patPFMetT1%s"),
   t01Variation = cms.InputTag("patPFMetT0pcT1"),
   t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%s"),
   
   tXYUncForRaw=cms.InputTag("patPFMetTxy"),
   tXYUncForT1=cms.InputTag("patPFMetT1Txy"),
   tXYUncForT01=cms.InputTag("patPFMetT1Txy"),
   tXYUncForT1Smear=cms.InputTag("patPFMetT1SmearTxy"),
   tXYUncForT01Smear=cms.InputTag("patPFMetT0T1SmearTxy"),
  
   #caloMET, will be used for the beginning of ata takin by the JetMET people
   caloMET = cms.InputTag("patCaloMet"),
)

