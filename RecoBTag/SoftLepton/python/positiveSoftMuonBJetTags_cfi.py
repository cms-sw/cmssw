import FWCore.ParameterSet.Config as cms

positiveSoftMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softMuonTagInfos"))
)
