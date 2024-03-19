import FWCore.ParameterSet.Config as cms

hltHpsPFTauDiscriminationByDecayModeFindingNewDMs = cms.EDProducer("PFRecoTauDiscriminationByHPSSelection",
    PFTauProducer = cms.InputTag("hltHpsPFTauProducer"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string('and')
    ),
    decayModes = cms.VPSet(
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(True),
                mass = cms.bool(True),
                phi = cms.bool(True)
            ),
            maxMass = cms.string('1.'),
            minMass = cms.double(-1000.0),
            nCharged = cms.uint32(1),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(0),
            nTracksMin = cms.uint32(1)
        ),
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(True),
                mass = cms.bool(True),
                phi = cms.bool(True)
            ),
            assumeStripMass = cms.double(0.1349),
            maxMass = cms.string('max(1.72, min(1.72*sqrt(pt/100.), 4.2))'),
            minMass = cms.double(0.0),
            nCharged = cms.uint32(1),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            nTracksMin = cms.uint32(1)
        ),
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(True),
                mass = cms.bool(True),
                phi = cms.bool(True)
            ),
            assumeStripMass = cms.double(0.0),
            maxMass = cms.string('max(1.72, min(1.72*sqrt(pt/100.), 4.0))'),
            maxPi0Mass = cms.double(0.8),
            minMass = cms.double(0.4),
            minPi0Mass = cms.double(0.0),
            nCharged = cms.uint32(1),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(2),
            nTracksMin = cms.uint32(1)
        ),
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(False),
                mass = cms.bool(True),
                phi = cms.bool(True)
            ),
            maxMass = cms.string('1.2'),
            minMass = cms.double(0.0),
            nCharged = cms.uint32(2),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(0),
            nTracksMin = cms.uint32(2)
        ),
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(False),
                mass = cms.bool(True),
                phi = cms.bool(True)
            ),
            maxMass = cms.string('max(1.6, min(1.6*sqrt(pt/100.), 4.0))'),
            minMass = cms.double(0.0),
            nCharged = cms.uint32(2),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            nTracksMin = cms.uint32(2)
        ),
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(False),
                mass = cms.bool(True),
                phi = cms.bool(True)
            ),
            maxMass = cms.string('1.6'),
            minMass = cms.double(0.7),
            nCharged = cms.uint32(3),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(0),
            nTracksMin = cms.uint32(2)
        ),
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(False),
                mass = cms.bool(False),
                phi = cms.bool(False)
            ),
            maxMass = cms.string('1.6'),
            minMass = cms.double(0.9),
            nCharged = cms.uint32(3),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            nTracksMin = cms.uint32(2)
        )
    ),
    matchingCone = cms.double(0.5),
    minPixelHits = cms.int32(0),
    minTauPt = cms.double(18.0),
    requireTauChargedHadronsToBeChargedPFCands = cms.bool(False),
    verbosity = cms.int32(0)
)
