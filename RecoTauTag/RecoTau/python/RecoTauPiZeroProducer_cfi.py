import FWCore.ParameterSet.Config as cms

import RecoTauTag.RecoTau.RecoTauPiZeroBuilderPlugins_cfi as builders
import RecoTauTag.RecoTau.RecoTauPiZeroQualityPlugins_cfi as ranking

ak5PFJetsRecoTauPiZeros = cms.EDProducer(
    "RecoTauPiZeroProducer",
    src = cms.InputTag("ak5PFJets"),
    massHypothesis = cms.double(0.136),
    builders = cms.VPSet(
        builders.combinatoricPhotonPairs,
        builders.strips,
    ),
    ranking = cms.VPSet(
        ranking.nearPiZeroMassBarrel, # Prefer pi zeros +- 0.05 GeV correct mass
        ranking.nearPiZeroMassEndcap,
        ranking.isInStrip,      # Allow incorrect masses if in strip
    ),
)

ak5PFJetsLegacyTaNCPiZeros = cms.EDProducer(
    "RecoTauPiZeroProducer",
    src = cms.InputTag("ak5PFJets"),
    massHypothesis = cms.double(0.136),
    builders = cms.VPSet(
        builders.allSinglePhotons,
        builders.combinatoricPhotonPairs,
    ),
    ranking = cms.VPSet(
        ranking.legacyPFTauDecayModeSelection
    ),
)

ak5PFJetsLegacyHPSPiZeros = cms.EDProducer(
    "RecoTauPiZeroProducer",
    src = cms.InputTag("ak5PFJets"),
    massHypothesis = cms.double(0.136),
    builders = cms.VPSet(
        builders.strips,
    ),
    ranking = cms.VPSet(
        ranking.isInStrip
    ),
)

