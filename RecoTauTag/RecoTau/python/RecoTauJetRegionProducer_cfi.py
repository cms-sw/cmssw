import FWCore.ParameterSet.Config as cms

RecoTauJetRegionProducer = cms.EDProducer(
        "RecoTauJetRegionProducer",
            deltaR = cms.double(0.8),
            src = cms.InputTag("ak4PFJets"),
            pfSrc = cms.InputTag("particleFlow"),
        )
