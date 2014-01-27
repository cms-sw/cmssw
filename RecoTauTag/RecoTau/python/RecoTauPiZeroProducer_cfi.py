import FWCore.ParameterSet.Config as cms

import RecoTauTag.RecoTau.RecoTauPiZeroBuilderPlugins_cfi as builders
import RecoTauTag.RecoTau.RecoTauPiZeroQualityPlugins_cfi as ranking

ak5PFJetsRecoTauGreedyPiZeros = cms.EDProducer(
    "RecoTauPiZeroProducer",
    jetSrc = cms.InputTag("ak5PFJets"),
    massHypothesis = cms.double(0.136),
    outputSelection = cms.string('pt > 1.5'),
    builders = cms.VPSet(
        builders.comboStrips
    ),
    ranking = cms.VPSet(
        ranking.greedy
    ),
)

ak5PFJetsRecoTauPiZeros = cms.EDProducer(
    "RecoTauPiZeroProducer",
    jetSrc = cms.InputTag("ak5PFJets"),
    massHypothesis = cms.double(0.136),
    outputSelection = cms.string('pt > 1.5'),
    builders = cms.VPSet(
        builders.combinatoricPhotonPairs,
        #builders.strips
        builders.modStrips 
    ),
    ranking = cms.VPSet(
        ranking.nearPiZeroMassBarrel, # Prefer pi zeros +- 0.05 GeV correct mass
        ranking.nearPiZeroMassEndcap,
        ranking.isInStrip             # Allow incorrect masses if in strip
    ),
)

ak5PFJetsLegacyTaNCPiZeros = cms.EDProducer(
    "RecoTauPiZeroProducer",
    jetSrc = cms.InputTag("ak5PFJets"),
    massHypothesis = cms.double(0.136),
    outputSelection = cms.string('pt > 1.5'),
    builders = cms.VPSet(
        builders.allSinglePhotons,
        builders.combinatoricPhotonPairs
    ),
    ranking = cms.VPSet(
        ranking.legacyPFTauDecayModeSelection
    ),
)

ak5PFJetsLegacyHPSPiZeros = cms.EDProducer(
    "RecoTauPiZeroProducer",
    jetSrc = cms.InputTag("ak5PFJets"),
    massHypothesis = cms.double(0.136),
    outputSelection = cms.string('pt > 0'),
    builders = cms.VPSet(
        #builders.strips
        builders.modStrips 
    ),
    ranking = cms.VPSet(
        ranking.isInStrip
    )
)

