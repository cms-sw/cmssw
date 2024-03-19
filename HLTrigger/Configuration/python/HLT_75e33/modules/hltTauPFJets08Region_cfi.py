import FWCore.ParameterSet.Config as cms

hltTauPFJets08Region = cms.EDProducer("RecoTauJetRegionProducer",
    deltaR = cms.double(0.8),
    maxJetAbsEta = cms.double(99.0),
    minJetPt = cms.double(-1.0),
    pfCandAssocMapSrc = cms.InputTag(""),
    pfCandSrc = cms.InputTag("particleFlowTmp"),
    src = cms.InputTag("hltAK4PFJets"),
    verbosity = cms.int32(0)
)
