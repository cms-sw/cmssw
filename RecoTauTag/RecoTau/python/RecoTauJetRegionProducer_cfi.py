import FWCore.ParameterSet.Config as cms

RecoTauJetRegionProducer = cms.EDProducer(
        "RecoTauJetRegionProducer",
            deltaR = cms.double(0.8),
            src = cms.InputTag("ak5PFJets"),
            pfCandSrc = cms.InputTag("particleFlow"),
            pfCandAssocMapSrc = cms.InputTag("")
        )
