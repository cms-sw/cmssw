import FWCore.ParameterSet.Config as cms

softMuonBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softMuonTagInfos"),
    jetTagComputer = cms.string('softMuon')
)


