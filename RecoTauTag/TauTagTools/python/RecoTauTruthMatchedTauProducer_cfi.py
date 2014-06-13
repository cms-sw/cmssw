import FWCore.ParameterSet.Config as cms
import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners
from RecoTauTag.RecoTau.RecoTauCleaner import RecoTauCleaner

trueRecoTaus = RecoTauCleaner.clone(
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

