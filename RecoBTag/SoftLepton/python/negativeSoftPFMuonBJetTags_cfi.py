import FWCore.ParameterSet.Config as cms

negativeSoftPFMuonBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftMuon'),
    tagInfos = cms.VInputTag(cms.InputTag("softPFMuonsTagInfos"))
)
