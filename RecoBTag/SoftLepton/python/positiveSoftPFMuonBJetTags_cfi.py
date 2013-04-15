import FWCore.ParameterSet.Config as cms

positiveSoftPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFLeptonsTagInfo","SPFMuons"))
)
