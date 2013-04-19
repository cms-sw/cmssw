import FWCore.ParameterSet.Config as cms
import RecoBTag.SoftLepton.muonSelection

softPFMuonsTagInfos = cms.EDProducer("SoftPFMuonTagInfoProducer",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak5PFJets"),
    MuonId =cms.int32(0)
)
