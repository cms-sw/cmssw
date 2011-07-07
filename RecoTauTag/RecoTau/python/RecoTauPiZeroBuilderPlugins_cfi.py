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
    qualityCuts = PFTauQualityCuts.signalQualityCuts,
)

# Produce a PiZero candidate for each possible photon pair
combinatoricPhotonPairs = cms.PSet(
    name = cms.string("2"),
    plugin = cms.string("RecoTauPiZeroCombinatoricPlugin"),
    qualityCuts = PFTauQualityCuts.signalQualityCuts,
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
    qualityCuts = PFTauQualityCuts.signalQualityCuts,
    primaryVertexSrc = cms.InputTag("offlinePrimaryVertices"),
    # Clusterize photons and electrons (PF numbering)
    stripCandidatesParticleIds   = cms.vint32(2, 4),
    stripEtaAssociationDistance  = cms.double(0.05),
    stripPhiAssociationDistance  = cms.double(0.2),
    makeCombinatoricStrips = cms.bool(False)
)

comboStrips = cms.PSet(
    name = cms.string("cs"),
    plugin = cms.string("RecoTauPiZeroStripPlugin"),
    qualityCuts = PFTauQualityCuts.signalQualityCuts,
    primaryVertexSrc = cms.InputTag("offlinePrimaryVertices"),
    # Clusterize photons and electrons (PF numbering)
    stripCandidatesParticleIds   = cms.vint32(2, 4),
    stripEtaAssociationDistance  = cms.double(0.05),
    stripPhiAssociationDistance  = cms.double(0.2),
    makeCombinatoricStrips = cms.bool(True),
    maxInputStrips = cms.int32(5),
    stripMassWhenCombining = cms.double(0.0), # assume photon like
)
