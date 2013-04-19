import FWCore.ParameterSet.Config as cms

softPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('softMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
