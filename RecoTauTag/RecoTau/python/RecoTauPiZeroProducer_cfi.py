import FWCore.ParameterSet.Config as cms

import RecoTauTag.RecoTau.RecoTauPiZeroBuilderPlugins_cfi as builders
import RecoTauTag.RecoTau.RecoTauPiZeroQualityPlugins_cfi as ranking

ak5PFJetsRecoTauPiZeros = cms.EDProducer(
    "RecoTauPiZeroProducer",
    jetSrc = cms.InputTag("ak5PFJets"),
    builders = cms.VPSet(
        builders.allSinglePhotons,
        builders.combinatoricPhotonPairs,
        builders.strips,
    ),
    ranking = cms.VPSet(
        ranking.nearPiZeroMass, # Prefer pi zeros +- 0.05 GeV correct mass
        ranking.isInStrip,      # Allow incorrect masses if in strip
        ranking.maximumMass,    # Set a maximum mass of 0.2 GeV
    ),
)
