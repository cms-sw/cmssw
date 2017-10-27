import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for reconstructing boosted taus using the HPS algorithm

'''

import CommonTools.ParticleFlow.pfNoPileUp_cff as boostedTaus
pfPileUpForBoostedTaus = boostedTaus.pfPileUp.clone(
    PFCandidates = cms.InputTag('particleFlow'),
    checkClosestZVertex = cms.bool(False)
)
pfNoPileUpForBoostedTaus = boostedTaus.pfNoPileUp.clone(
    topCollection = cms.InputTag('pfPileUpForBoostedTaus'),
    bottomCollection = cms.InputTag('particleFlow')
)


import RecoJets.JetProducers.ak4PFJets_cfi as boostedTaus2
import RecoJets.JetProducers.CMSBoostedTauSeedingParameters_cfi as boostedTaus3
ca8PFJetsCHSprunedForBoostedTaus = boostedTaus2.ak4PFJets.clone(
    boostedTaus3.CMSBoostedTauSeedingParameters,
    #src = cms.InputTag('pfNoPileUpForBoostedTaus'),
    jetPtMin = cms.double(100.0),
    doAreaFastjet = cms.bool(True),
    nFilt = cms.int32(100),
    rParam = cms.double(0.8),
    jetAlgorithm = cms.string("CambridgeAachen"),
    writeCompound = cms.bool(True),
    jetCollInstanceName = cms.string('subJetsForSeedingBoostedTaus')
)

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
pp_on_XeXe_2017.toModify(ca8PFJetsCHSprunedForBoostedTaus, inputEtMin = 999999.0)

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
