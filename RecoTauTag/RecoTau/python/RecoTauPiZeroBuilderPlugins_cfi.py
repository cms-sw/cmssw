import FWCore.ParameterSet.Config as cms

'''

Configuration for Pi Zero producer plugins.

Author: Evan K. Friis, UC Davis


'''

# Produce a PiZero candidate for each photon - the "trivial" case
allSinglePhotons = cms.PSet(
    name = cms.string("SinglePhotons"),
    plugin = cms.string("RecoTauPiZeroTrivialPlugin"),
)

# Produce a PiZero candidate for each possible photon pair
combinatoricPhotonPairs = cms.PSet(
    name = cms.string("Combinatoric"),
    plugin = cms.string("RecoTauPiZeroCombinatoricPlugin"),
    # Determine the maximum number of PiZeros to use. -1 for all
    maxInputGammas = cms.uint32(10),
    # Mass constraints taken care of during cleaning.
    minMass = cms.double(0.0),
    maxMass = cms.double(-1.0),
    choose = cms.uint32(2),
)

# Produce a "strips" of photons
strips = cms.PSet(
    name = cms.string("Strips"),
    plugin = cms.string("RecoTauPiZeroStripPlugin"),
    stripCandidatesParticleIds   = cms.vint32(2, 4),         #Clusterize photons and electrons (PF numbering)
    stripEtaAssociationDistance  = cms.double(0.05),         #Eta Association for the strips
    stripPhiAssociationDistance  = cms.double(0.2),          #Phi Association for the strips
)

