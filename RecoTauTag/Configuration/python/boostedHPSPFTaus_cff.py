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

##import RecoJets.JetProducers.ak5PFJetsPruned_cfi as boostedTaus2
import RecoJets.JetProducers.ak5PFJets_cfi as boostedTaus2
import RecoJets.JetProducers.CMSBoostedTauSeedingParameters_cfi as boostedTaus3
##ca8PFJetsCHSprunedForBoostedTaus = boostedTaus2.ak5PFJetsPruned.clone(
ca8PFJetsCHSprunedForBoostedTaus = boostedTaus2.ak5PFJets.clone(
    boostedTaus3.CMSBoostedTauSeedingParameters,
    src = cms.InputTag('pfNoPileUpForBoostedTaus'),
    jetPtMin = cms.double(10.0),
    doAreaFastjet = cms.bool(True),
    nFilt = cms.int32(4),
    rParam = cms.double(0.8),
    jetAlgorithm = cms.string("CambridgeAachen"),
    writeCompound = cms.bool(True),
    jetCollInstanceName = cms.string('subJetsForSeedingBoostedTaus')
)

boostedTauSeeds = cms.EDProducer("BoostedTauSeedsProducer",
    subjetSrc = cms.InputTag('ca8PFJetsCHSprunedForBoostedTaus', 'subJetsForSeedingBoostedTaus'),
    pfCandidateSrc = cms.InputTag('particleFlow'),
    verbosity = cms.int32(0)
)

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
recoTauAK5PFJets08Region.src = cms.InputTag('boostedTauSeeds')
recoTauAK5PFJets08Region.pfCandSrc = cms.InputTag('pfNoPileUpForBoostedTaus')
recoTauAK5PFJets08Region.pfCandAssocMapSrc = cms.InputTag('boostedTauSeeds', 'pfCandAssocMapForIsolation')

ak5PFJetsLegacyHPSPiZeros.jetSrc = cms.InputTag('boostedTauSeeds')

ak5PFJetsRecoTauChargedHadrons.jetSrc = cms.InputTag('boostedTauSeeds')
ak5PFJetsRecoTauChargedHadrons.builders[1].dRcone = cms.double(0.3)
ak5PFJetsRecoTauChargedHadrons.builders[1].dRconeLimitedToJetArea = cms.bool(True)

combinatoricRecoTaus.jetSrc = cms.InputTag('boostedTauSeeds')
combinatoricRecoTaus.builders[0].pfCandSrc = cms.InputTag('pfNoPileUpForBoostedTaus')
combinatoricRecoTaus.modifiers.remove(combinatoricRecoTaus.modifiers[3])

hpsPFTauDiscriminationByLooseMuonRejection3.dRmuonMatch = cms.double(0.3)
hpsPFTauDiscriminationByLooseMuonRejection3.dRmuonMatchLimitedToJetArea = cms.bool(True)
hpsPFTauDiscriminationByTightMuonRejection3.dRmuonMatch = cms.double(0.3)
hpsPFTauDiscriminationByTightMuonRejection3.dRmuonMatchLimitedToJetArea = cms.bool(True)

produceAndDiscriminateBoostedHPSPFTaus = cms.Sequence(
    pfPileUpForBoostedTaus*
    pfNoPileUpForBoostedTaus*
    ca8PFJetsCHSprunedForBoostedTaus*
    boostedTauSeeds*
    PFTau
)    

