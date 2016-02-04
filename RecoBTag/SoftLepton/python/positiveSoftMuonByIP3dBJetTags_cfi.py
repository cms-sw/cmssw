import FWCore.ParameterSet.Config as cms

positiveSoftMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('positiveSoftLeptonByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softMuonTagInfos"))
)
