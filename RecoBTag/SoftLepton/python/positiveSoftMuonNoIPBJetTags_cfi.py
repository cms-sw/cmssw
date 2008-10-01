import FWCore.ParameterSet.Config as cms

positiveSoftMuonNoIPBJetTags = cms.EDProducer("JetTagProducer",
    tagInfos = cms.VInputTag( cms.InputTag("softMuonTagInfos") ),
    jetTagComputer = cms.string('positiveSoftMuonNoIP')
)
