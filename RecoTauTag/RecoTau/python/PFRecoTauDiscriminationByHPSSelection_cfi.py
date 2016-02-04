import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

hpsSelectionDiscriminator = cms.EDProducer(
    "PFRecoTauDiscriminationByHPSSelection",
    PFTauProducer = cms.InputTag('combinatoricRecoTaus'),
    Prediscriminants = noPrediscriminants,
    matchingCone = cms.double(0.1),
    minTauPt = cms.double(15.),
    coneSizeFormula = cms.string("max(min(0.1, 2.8/et()),0.05)"),
    decayModes = cms.VPSet(
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(0),
            # Always passes
            # If an PF electron is selected as the lead track, the tau can have
            # negative mass. FIXME - investigate this
            minMass = cms.double(-0.1),
            maxMass = cms.double(1),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            minMass = cms.double(0.3),
            maxMass = cms.double(1.3),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(2),
            minMass = cms.double(0.4),
            maxMass = cms.double(1.2),
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(0),
            minMass = cms.double(0.8),
            maxMass = cms.double(1.5),
        ),
    )
)



