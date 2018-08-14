import FWCore.ParameterSet.Config as cms

slimmedGenJetsFlavourInfos = cms.EDProducer("GenJetFlavourInfoPreserver",
    genJets = cms.InputTag("ak4GenJetsNoNu"),
    slimmedGenJets = cms.InputTag("slimmedGenJets"),                                      
    genJetFlavourInfos = cms.InputTag("ak4GenJetFlavourInfos"),
    slimmedGenJetAssociation = cms.InputTag("slimmedGenJets", "slimmedGenJetAssociation")
)
