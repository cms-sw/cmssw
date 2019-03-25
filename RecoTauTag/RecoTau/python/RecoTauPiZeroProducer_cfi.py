import FWCore.ParameterSet.Config as cms

import RecoTauTag.RecoTau.RecoTauPiZeroBuilderPlugins_cfi as builders
import RecoTauTag.RecoTau.RecoTauPiZeroQualityPlugins_cfi as ranking
from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common


from RecoTauTag.RecoTau.ak4PFJetsLegacyHPSPiZeros_cfi import ak4PFJetsLegacyHPSPiZeros
ak4PFJetsLegacyHPSPiZeros = ak4PFJetsLegacyHPSPiZeros.clone(
    jetSrc = PFRecoTauPFJetInputs.inputJetCollection,
    minJetPt = PFRecoTauPFJetInputs.minJetPt,
    maxJetAbsEta = PFRecoTauPFJetInputs.maxJetAbsEta,
    massHypothesis = cms.double(0.136),
    outputSelection = cms.string('pt > 0'),
    builders = cms.VPSet(
        #builders.strips
        #builders.modStrips
        builders.modStrips2
    ),
    ranking = cms.VPSet(
        ranking.isInStrip
    ),
    verbosity = cms.int32(0)
)
phase2_common.toModify(ak4PFJetsLegacyHPSPiZeros, 
                       builders = cms.VPSet(builders.modStrips) )

from RecoTauTag.RecoTau.ak4PFJetsRecoTauGreedyPiZeros_cfi import ak4PFJetsRecoTauGreedyPiZeros
ak4PFJetsRecoTauGreedyPiZeros = ak4PFJetsRecoTauGreedyPiZeros.clone( 
    jetSrc = PFRecoTauPFJetInputs.inputJetCollection,
    minJetPt = PFRecoTauPFJetInputs.minJetPt,
    maxJetAbsEta = PFRecoTauPFJetInputs.maxJetAbsEta,
    massHypothesis = cms.double(0.136),
    outputSelection = cms.string('pt > 1.5'),
    builders = cms.VPSet(
        builders.comboStrips
    ),
    ranking = cms.VPSet(
        ranking.greedy
    ),
)

from RecoTauTag.RecoTau.ak4PFJetsRecoTauPiZeros_cfi import ak4PFJetsRecoTauPiZeros
ak4PFJetsRecoTauPiZeros = ak4PFJetsRecoTauPiZeros.clone(
    jetSrc = PFRecoTauPFJetInputs.inputJetCollection,
    minJetPt = PFRecoTauPFJetInputs.minJetPt,
    maxJetAbsEta = PFRecoTauPFJetInputs.maxJetAbsEta,
    massHypothesis = cms.double(0.136),
    outputSelection = cms.string('pt > 1.5'),
    builders = cms.VPSet(
        builders.combinatoricPhotonPairs,
        #builders.strips
        #builders.modStrips
        builders.modStrips2
    ),
    ranking = cms.VPSet(
        ranking.nearPiZeroMassBarrel, # Prefer pi zeros +- 0.05 GeV correct mass
        ranking.nearPiZeroMassEndcap,
        ranking.isInStrip             # Allow incorrect masses if in strip
    ),
)

from RecoTauTag.RecoTau.ak4PFJetsLegacyTaNCPiZeros_cfi import ak4PFJetsLegacyTaNCPiZeros
ak4PFJetsLegacyTaNCPiZeros = ak4PFJetsLegacyHPSPiZeros.clone(
    jetSrc = PFRecoTauPFJetInputs.inputJetCollection,
    minJetPt = PFRecoTauPFJetInputs.minJetPt,
    maxJetAbsEta = PFRecoTauPFJetInputs.maxJetAbsEta,
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
