import FWCore.ParameterSet.Config as cms

softMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softMuonTagInfos"))
)
