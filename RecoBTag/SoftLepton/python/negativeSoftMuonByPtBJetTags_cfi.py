import FWCore.ParameterSet.Config as cms

negativeSoftLeptonByPtBJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('negativeSoftLeptonByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("softMuonTagInfos"))
)
