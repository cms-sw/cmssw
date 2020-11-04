import FWCore.ParameterSet.Config as cms

'''

Sequences for reconstructing boosted taus using the HPS algorithm

'''

import CommonTools.ParticleFlow.pfNoPileUp_cff as boostedTaus
pfPileUpForBoostedTaus = boostedTaus.pfPileUp.clone(
    PFCandidates = 'particleFlow',
    checkClosestZVertex = False
)
pfNoPileUpForBoostedTaus = boostedTaus.pfNoPileUp.clone(
    topCollection = 'pfPileUpForBoostedTaus',
    bottomCollection = 'particleFlow'
)


import RecoJets.JetProducers.ak4PFJets_cfi as boostedTaus2
import RecoJets.JetProducers.CMSBoostedTauSeedingParameters_cfi as boostedTaus3
ca8PFJetsCHSprunedForBoostedTaus = boostedTaus2.ak4PFJets.clone(
    boostedTaus3.CMSBoostedTauSeedingParameters,
    #src = 'pfNoPileUpForBoostedTaus',
    jetPtMin = 100.0,
    doAreaFastjet = True,
    nFilt = cms.int32(100),
    rParam = 0.8,
    jetAlgorithm = "CambridgeAachen",
    writeCompound = cms.bool(True),
    jetCollInstanceName = cms.string('subJetsForSeedingBoostedTaus')
)

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(ca8PFJetsCHSprunedForBoostedTaus, inputEtMin = 999999.0, src = "particleFlow")

boostedTauSeeds = cms.EDProducer("BoostedTauSeedsProducer",
    subjetSrc = cms.InputTag('ca8PFJetsCHSprunedForBoostedTaus', 'subJetsForSeedingBoostedTaus'),
    pfCandidateSrc = cms.InputTag('particleFlow'),
    verbosity = cms.int32(0)
)

boostedHPSPFTausTask = cms.Task(
    pfPileUpForBoostedTaus,
    pfNoPileUpForBoostedTaus,
    ca8PFJetsCHSprunedForBoostedTaus,
    boostedTauSeeds
)
