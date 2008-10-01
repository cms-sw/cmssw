import FWCore.ParameterSet.Config as cms

negativeSoftMuonNoIPBJetTags = cms.EDProducer("JetTagProducer",
    tagInfos = cms.VInputTag( cms.InputTag("softMuonTagInfos") ),
    jetTagComputer = cms.string('negativeSoftMuonNoIP')
)
