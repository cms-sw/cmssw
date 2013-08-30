import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

decayMode_1Prong0Pi0 = cms.PSet(
    nCharged = cms.uint32(1),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(1),
    nChargedPFCandsMin = cms.uint32(1),
    # Always passes
    # If an PF electron is selected as the lead track, the tau can have
    # negative mass. FIXME - investigate this
    minMass = cms.double(-1.e3),
    maxMass = cms.string("1.")
)
decayMode_1Prong1Pi0 = cms.PSet(
    nCharged = cms.uint32(1),
    nPiZeros = cms.uint32(1),
    nTracksMin = cms.uint32(1),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.3),
    maxMass = cms.string("max(1.3, min(1.3*sqrt(pt/100.), 4.2))"),
    assumeStripMass = cms.double(0.1349)
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
    assumeStripMass = cms.double(0.0)
)
decayMode_2Prong0Pi0 = cms.PSet(
    nCharged = cms.uint32(2),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(2),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.),
    maxMass = cms.string("1.2")
)
decayMode_2Prong1Pi0 = cms.PSet(
    nCharged = cms.uint32(2),
    nPiZeros = cms.uint32(1),
    nTracksMin = cms.uint32(2),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.),
    maxMass = cms.string("max(1.2, min(1.2*sqrt(pt/100.), 4.0))")
)
decayMode_3Prong0Pi0 = cms.PSet(
    nCharged = cms.uint32(3),
    nPiZeros = cms.uint32(0),
    nTracksMin = cms.uint32(2),
    nChargedPFCandsMin = cms.uint32(1),
    minMass = cms.double(0.8),
    maxMass = cms.string("1.5")
)

hpsSelectionDiscriminator = cms.EDProducer(
    "PFRecoTauDiscriminationByHPSSelection",
    PFTauProducer = cms.InputTag('combinatoricRecoTaus'),
    Prediscriminants = noPrediscriminants,
    matchingCone = cms.double(0.5),
    minTauPt = cms.double(0.0),
    coneSizeFormula = cms.string("max(min(0.1, 3.0/pt()), 0.05)"),
    decayModes = cms.VPSet(
        decayMode_1Prong0Pi0,
        decayMode_1Prong1Pi0,
        decayMode_1Prong2Pi0,
        decayMode_2Prong0Pi0,
        decayMode_2Prong1Pi0,
        decayMode_3Prong0Pi0
    ),
    requireTauChargedHadronsToBeChargedPFCands = cms.bool(False)
)



