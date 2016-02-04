import FWCore.ParameterSet.Config as cms

softMuonByIP3dBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softLeptonByIP3d'),
    tagInfos = cms.VInputTag(cms.InputTag("softMuonTagInfos"))
)
