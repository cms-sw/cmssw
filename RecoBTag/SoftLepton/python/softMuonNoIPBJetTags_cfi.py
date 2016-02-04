import FWCore.ParameterSet.Config as cms

softMuonNoIPBJetTags = cms.EDProducer("JetTagProducer",
    tagInfos = cms.VInputTag( cms.InputTag("softMuonTagInfos") ),
    jetTagComputer = cms.string('softMuonNoIP')
)
