import FWCore.ParameterSet.Config as cms

slimmedMETs = cms.EDProducer("PATMETSlimmer",
   src = cms.InputTag("patMETs"),                
   rawVariation   = cms.InputTag("patPFMet"),
   t1Uncertainties  = cms.InputTag("patPFMetT1%s"),
   t01Variation = cms.InputTag("patPFMetT0pcT1"),
   t1SmearedVarsAndUncs = cms.InputTag("patPFMetT1Smear%s"),
   
   tXYUncForRaw=cms.InputTag("patPFMetTxy"),
   tXYUncForT1=cms.InputTag("patPFMetT1Txy"),
   tXYUncForT01=cms.InputTag("patPFMetT0pcT1Txy"),
   tXYUncForT1Smear=cms.InputTag("patPFMetT1SmearTxy"),
   tXYUncForT01Smear=cms.InputTag("patPFMetT0pcT1SmearTxy"),
  
   #caloMET, will be used for the beginning of ata takin by the JetMET people
   caloMET = cms.InputTag("patCaloMet"),

   #adding CHS and Track MET for the Jet/MET studies
   chsMET = cms.InputTag("patCHSMet"),
   trkMET = cms.InputTag("patTrkMet"),

   #switch to read the type0 correction from the existing slimmedMET
   #when running on top of miniAOD (type0 cannot be redone at the miniAOD level
   runningOnMiniAOD = cms.bool(False)

)

