import FWCore.ParameterSet.Config as cms

'''

Configuration for Pi Zero producer plugins.

Author: Evan K. Friis, UC Davis


'''
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts

# Produce a PiZero candidate for each photon - the "trivial" case
allSinglePhotons = cms.PSet(
    name = cms.string("1"),
    plugin = cms.string("RecoTauPiZeroTrivialPlugin"),
    qualityCuts = PFTauQualityCuts,
)

# Produce a PiZero candidate for each possible photon pair
combinatoricPhotonPairs = cms.PSet(
    name = cms.string("2"),
    plugin = cms.string("RecoTauPiZeroCombinatoricPlugin"),
    qualityCuts = PFTauQualityCuts,
    # Determine the maximum number of PiZeros to use. -1 for all
    maxInputGammas = cms.uint32(10),
    # Mass constraints taken care of during cleaning.
    minMass = cms.double(0.0),
    maxMass = cms.double(-1.0),
    choose = cms.uint32(2),
)

# Produce a "strips" of photons
strips = cms.PSet(
    name = cms.string("s"),
    plugin = cms.string("RecoTauPiZeroStripPlugin"),
    qualityCuts = PFTauQualityCuts,
    # Clusterize photons and electrons (PF numbering)
    stripCandidatesParticleIds  = cms.vint32(2, 4),
    stripEtaAssociationDistance = cms.double(0.05),
    stripPhiAssociationDistance = cms.double(0.2),
    makeCombinatoricStrips = cms.bool(False),
    verbosity = cms.int32(0)
)

comboStrips = strips.clone(
    name = "cs",
    plugin = "RecoTauPiZeroStripPlugin",
    makeCombinatoricStrips = True,
    maxInputStrips = cms.int32(5),
    stripMassWhenCombining = cms.double(0.0), # assume photon like
)

# Produce a "strips" of photons
# with no track quality cuts applied to PFElectrons
modStrips = strips.clone(
    plugin = 'RecoTauPiZeroStripPlugin2',
    applyElecTrackQcuts = cms.bool(False),
    minGammaEtStripSeed = cms.double(1.0),
    minGammaEtStripAdd = cms.double(1.0),
    minStripEt = cms.double(1.0),
    updateStripAfterEachDaughter = cms.bool(False),
    maxStripBuildIterations = cms.int32(-1),
)

# Produce a "strips" of photons
# with no track quality cuts applied to PFElectrons
# and eta x phi size of strip increasing for low pT photons

modStrips2 = strips.clone(
    plugin = 'RecoTauPiZeroStripPlugin3',
    qualityCuts = PFTauQualityCuts,
    applyElecTrackQcuts = cms.bool(False),
    stripEtaAssociationDistanceFunc = cms.PSet(
        function = cms.string("TMath::Min(0.15, TMath::Max(0.05, [0]*TMath::Power(pT, -[1])))"),
        par0 = cms.double(1.97077e-01),
        par1 = cms.double(6.58701e-01)
    ),
    stripPhiAssociationDistanceFunc = cms.PSet(
        function = cms.string("TMath::Min(0.3, TMath::Max(0.05, [0]*TMath::Power(pT, -[1])))"),
        par0 = cms.double(3.52476e-01),
        par1 = cms.double(7.07716e-01)
    ),
    makeCombinatoricStrips = False,
    minGammaEtStripSeed = cms.double(1.0),
    minGammaEtStripAdd = cms.double(1.0),
    minStripEt = cms.double(1.0),
    # CV: parametrization of strip size in eta and phi determined by Yuta Takahashi,
    #     chosen to contain 95% of photons from tau decays
    updateStripAfterEachDaughter = cms.bool(False),
    maxStripBuildIterations = cms.int32(-1),
)
