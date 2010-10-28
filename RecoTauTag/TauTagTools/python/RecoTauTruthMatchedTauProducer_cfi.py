import FWCore.ParameterSet.Config as cms
import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners

trueRecoTaus = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("combinatoricRecoTaus"),
    cleaners = cms.VPSet(
        cms.PSet(
            name = cms.string("TruthDecayModeMatch"),
            plugin = cms.string("RecoTauDecayModeTruthMatchPlugin"),
            matching = cms.InputTag("recoTauTruthMatcher"),
        ),
        cms.PSet(
            name = cms.string("TruthPtMatch"),
            plugin = cms.string("RecoTauDistanceFromTruthPlugin"),
            matching = cms.InputTag("recoTauTruthMatcher"),
        ),
        cleaners.tanc.clone(
            src = cms.InputTag("combinatoricRecoTausDiscriminationByTaNC"),
        ),
    )
)

