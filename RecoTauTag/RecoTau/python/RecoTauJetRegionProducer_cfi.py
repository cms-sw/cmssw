import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs

RecoTauJetRegionProducer = cms.EDProducer(
        "RecoTauJetRegionProducer",
            deltaR = cms.double(0.8),
            src = PFRecoTauPFJetInputs.inputJetCollection,
            minJetPt = PFRecoTauPFJetInputs.minJetPt,
            maxJetAbsEta = PFRecoTauPFJetInputs.maxJetAbsEta,
            pfCandSrc = cms.InputTag("particleFlow"),
            pfCandAssocMapSrc = cms.InputTag("")
        )
