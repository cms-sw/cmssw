import FWCore.ParameterSet.Config as cms

softMuonNoIPBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softMuonTagInfos"),
    jetTagComputer = cms.string('softMuonNoIP')
)


