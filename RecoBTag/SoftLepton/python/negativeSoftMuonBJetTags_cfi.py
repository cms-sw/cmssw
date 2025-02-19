import FWCore.ParameterSet.Config as cms

negativeSoftMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softMuonTagInfos"))
)
