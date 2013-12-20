import FWCore.ParameterSet.Config as cms

recoTauAK5PFJets08Region = cms.EDProducer(
        "RecoTauJetRegionProducer",
            deltaR = cms.double(0.8),
            src = cms.InputTag("ak5PFJets"),
            pfCandSrc = cms.InputTag("particleFlow"),
            pfCandAssocMapSrc = cms.InputTag("")
        )
