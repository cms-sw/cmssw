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
        ranking.nearPiZeroMassBarrel, # Prefer pi zeros +- 0.05 GeV correct mass
        ranking.nearPiZeroMassEndcap,
        ranking.isInStrip,      # Allow incorrect masses if in strip
    ),
)
