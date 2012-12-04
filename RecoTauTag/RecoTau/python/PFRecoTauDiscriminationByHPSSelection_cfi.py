import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

hpsSelectionDiscriminator = cms.EDProducer(
    "PFRecoTauDiscriminationByHPSSelection",
    PFTauProducer = cms.InputTag('combinatoricRecoTaus'),
    Prediscriminants = noPrediscriminants,
    matchingCone = cms.double(0.5),
    minTauPt = cms.double(0.0),
    coneSizeFormula = cms.string("max(min(0.1, 3.5/pt()),0.05)"),
    decayModes = cms.VPSet(
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(0),
            # Always passes
            # If an PF electron is selected as the lead track, the tau can have
            # negative mass. FIXME - investigate this
            minMass = cms.double(-1.e3),
            maxMass = cms.string("1.")
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            minMass = cms.double(0.3),
            maxMass = cms.string("max(1.3, min(1.3*sqrt(pt/200.), 2.1))"),
            assumeStripMass = cms.double(0.1349)
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(2),
            minMass = cms.double(0.4),
            maxMass = cms.string("max(1.2, min(1.2*sqrt(pt/200.), 2.0))"),
            minPi0Mass = cms.double(0.05),
            maxPi0Mass = cms.double(0.2),
            # Here the strips are assumed to correspond to photons
            assumeStripMass = cms.double(0.0)
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(0),
            minMass = cms.double(0.6),
            maxMass = cms.string("1.7")
        )
    )
)



