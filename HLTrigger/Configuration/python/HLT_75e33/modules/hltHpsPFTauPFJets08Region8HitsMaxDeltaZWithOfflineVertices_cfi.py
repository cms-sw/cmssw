import FWCore.ParameterSet.Config as cms

hltHpsPFTauPFJets08Region8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("RecoTauJetRegionProducer",
    deltaR = cms.double(0.8),
    maxJetAbsEta = cms.double(4.0),
    minJetPt = cms.double(14.0),
    pfCandAssocMapSrc = cms.InputTag(""),
    pfCandSrc = cms.InputTag("particleFlowTmp"),
    src = cms.InputTag("hltHpsPFTauAK4PFJets8HitsMaxDeltaZWithOfflineVertices"),
    verbosity = cms.int32(0)
)
