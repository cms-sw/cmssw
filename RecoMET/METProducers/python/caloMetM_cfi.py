import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
caloMetM = cms.EDProducer(
    "MuonMET",
    metTypeInputTag = cms.InputTag("CaloMET"),
    uncorMETInputTag = cms.InputTag("caloMet"),
    muonsInputTag  = cms.InputTag("muons"),
    muonMETDepositValueMapInputTag = cms.InputTag("muonMETValueMapProducer", "muCorrData", "")
    )

##____________________________________________________________________________||
