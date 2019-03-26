import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs

minPi0Mass_default = -1.e3
maxPi0Mass_default = 1.e9
assumeStripMass_default = -1.0

decayMode_1Prong0Pi0 = cms.PSet(
    nCharged = cms.uint32(1),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(1),
    nChargedPFCandsMin = cms.uint32(1),
    # Always passes
    # If an PF electron is selected as the lead track, the tau can have
    # negative mass. FIXME - investigate this
    minMass = cms.double(-1.e3),
    maxMass = cms.string("1."),
    minPi0Mass = cms.double(minPi0Mass_default),
    maxPi0Mass = cms.double(maxPi0Mass_default),
    # for XProng0Pi0 decay modes bending corrections are transparent
    assumeStripMass = cms.double(assumeStripMass_default),
    applyBendCorrection = cms.PSet(
        eta = cms.bool(True),
        phi = cms.bool(True),
        mass = cms.bool(True)
    )
)
decayMode_1Prong1Pi0 = cms.PSet(
    nCharged = cms.uint32(1),
    nPiZeros = cms.uint32(1),
    nTracksMin = cms.uint32(1),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.3),
    maxMass = cms.string("max(1.3, min(1.3*sqrt(pt/100.), 4.2))"),
    minPi0Mass = cms.double(minPi0Mass_default),
    maxPi0Mass = cms.double(maxPi0Mass_default),
    assumeStripMass = cms.double(0.1349),
    applyBendCorrection = cms.PSet(
        eta = cms.bool(True),
        phi = cms.bool(True),
        mass = cms.bool(True)
    )
)
decayMode_1Prong2Pi0 = cms.PSet(
    nCharged = cms.uint32(1),
    nPiZeros = cms.uint32(2),
    nTracksMin = cms.uint32(1),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.4),
    maxMass = cms.string("max(1.2, min(1.2*sqrt(pt/100.), 4.0))"),
    minPi0Mass = cms.double(0.05),
    maxPi0Mass = cms.double(0.2),
    # Here the strips are assumed to correspond to photons
    assumeStripMass = cms.double(0.0),
    applyBendCorrection = cms.PSet(
        eta = cms.bool(True),
        phi = cms.bool(True),
        mass = cms.bool(True)
    )
)
decayMode_2Prong0Pi0 = cms.PSet(
    nCharged = cms.uint32(2),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(2),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.),
    maxMass = cms.string("1.2"),
    minPi0Mass = cms.double(minPi0Mass_default),
    maxPi0Mass = cms.double(maxPi0Mass_default),
    # for XProng0Pi0 decay modes bending corrections are transparent
    assumeStripMass = cms.double(assumeStripMass_default),
    applyBendCorrection = cms.PSet(
        eta = cms.bool(False),
        phi = cms.bool(False),
        mass = cms.bool(False)
    )
)
decayMode_2Prong1Pi0 = cms.PSet(
    nCharged = cms.uint32(2),
    nPiZeros = cms.uint32(1),
    nTracksMin = cms.uint32(2),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.),
    maxMass = cms.string("max(1.2, min(1.2*sqrt(pt/100.), 4.0))"),
    minPi0Mass = cms.double(minPi0Mass_default),
    maxPi0Mass = cms.double(maxPi0Mass_default),
    assumeStripMass = cms.double(assumeStripMass_default),
    applyBendCorrection = cms.PSet(
        eta = cms.bool(False),
        phi = cms.bool(False),
        mass = cms.bool(False)
    )
)
decayMode_3Prong0Pi0 = cms.PSet(
    nCharged = cms.uint32(3),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(2),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.8),
    maxMass = cms.string("1.5"),
    minPi0Mass = cms.double(minPi0Mass_default),
    maxPi0Mass = cms.double(maxPi0Mass_default),
    assumeStripMass = cms.double(assumeStripMass_default),
    applyBendCorrection = cms.PSet(
        eta = cms.bool(False),
        phi = cms.bool(False),
        mass = cms.bool(False)
    )
)
decayMode_3Prong1Pi0 = cms.PSet( #suggestions made by CV
    nCharged = cms.uint32(3),
    nPiZeros = cms.uint32(1),
    nTracksMin = cms.uint32(2),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.9),
    maxMass = cms.string("1.6"),
    minPi0Mass = cms.double(minPi0Mass_default),
    maxPi0Mass = cms.double(maxPi0Mass_default),
    # for XProng0Pi0 decay modes bending corrections are transparent
    assumeStripMass = cms.double(assumeStripMass_default),
    applyBendCorrection = cms.PSet(
        eta = cms.bool(False),
        phi = cms.bool(False),
        mass = cms.bool(False)
    )
)

hpsSelectionDiscriminator = cms.EDProducer(
    "PFRecoTauDiscriminationByHPSSelection",
    PFTauProducer = cms.InputTag('combinatoricRecoTaus'),
    Prediscriminants = noPrediscriminants,
    matchingCone = PFRecoTauPFJetInputs.jetConeSize,
    minTauPt = cms.double(0.0),
    decayModes = cms.VPSet(
        decayMode_1Prong0Pi0,
        decayMode_1Prong1Pi0,
        decayMode_1Prong2Pi0,
        decayMode_2Prong0Pi0,
        decayMode_2Prong1Pi0,
        decayMode_3Prong0Pi0,
	decayMode_3Prong1Pi0
    ),
    requireTauChargedHadronsToBeChargedPFCands = cms.bool(False),
    # CV: require at least one pixel hit for the sum of all tracks
    minPixelHits = cms.int32(1),
    verbosity = cms.int32(0)
)



