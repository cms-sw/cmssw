import FWCore.ParameterSet.Config as cms

hltHpsPFTauDiscriminationByDecayModeFindingNewDMs8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("PFRecoTauDiscriminationByHPSSelection",
    PFTauProducer = cms.InputTag("hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices"),
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
            assumeStripMass = cms.double(-1.0),
            maxMass = cms.string('1.'),
            maxPi0Mass = cms.double(1000000000.0),
            minMass = cms.double(-1000.0),
            minPi0Mass = cms.double(-1000.0),
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
            maxMass = cms.string('max(1.3, min(1.3*sqrt(pt/100.), 4.2))'),
            maxPi0Mass = cms.double(1000000000.0),
            minMass = cms.double(0.3),
            minPi0Mass = cms.double(-1000.0),
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
            maxMass = cms.string('max(1.2, min(1.2*sqrt(pt/100.), 4.0))'),
            maxPi0Mass = cms.double(0.2),
            minMass = cms.double(0.4),
            minPi0Mass = cms.double(0.05),
            nCharged = cms.uint32(1),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(2),
            nTracksMin = cms.uint32(1)
        ),
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(False),
                mass = cms.bool(False),
                phi = cms.bool(False)
            ),
            assumeStripMass = cms.double(-1.0),
            maxMass = cms.string('1.2'),
            maxPi0Mass = cms.double(1000000000.0),
            minMass = cms.double(0.0),
            minPi0Mass = cms.double(-1000.0),
            nCharged = cms.uint32(2),
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
            assumeStripMass = cms.double(-1.0),
            maxMass = cms.string('max(1.2, min(1.2*sqrt(pt/100.), 4.0))'),
            maxPi0Mass = cms.double(1000000000.0),
            minMass = cms.double(0.0),
            minPi0Mass = cms.double(-1000.0),
            nCharged = cms.uint32(2),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            nTracksMin = cms.uint32(2)
        ),
        cms.PSet(
            applyBendCorrection = cms.PSet(
                eta = cms.bool(False),
                mass = cms.bool(False),
                phi = cms.bool(False)
            ),
            assumeStripMass = cms.double(-1.0),
            maxMass = cms.string('1.5'),
            maxPi0Mass = cms.double(1000000000.0),
            minMass = cms.double(0.8),
            minPi0Mass = cms.double(-1000.0),
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
            assumeStripMass = cms.double(-1.0),
            maxMass = cms.string('1.6'),
            maxPi0Mass = cms.double(1000000000.0),
            minMass = cms.double(0.9),
            minPi0Mass = cms.double(-1000.0),
            nCharged = cms.uint32(3),
            nChargedPFCandsMin = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            nTracksMin = cms.uint32(2)
        )
    ),
    matchingCone = cms.double(0.5),
    minPixelHits = cms.int32(1),
    minTauPt = cms.double(0.0),
    requireTauChargedHadronsToBeChargedPFCands = cms.bool(True),
    verbosity = cms.int32(0)
)
