import FWCore.ParameterSet.Config as cms

muonSeededSeedsOutInDisplaced = cms.EDProducer("OutsideInMuonSeeder",
    cut = cms.string('pt > 10 && outerTrack.hitPattern.muonStationsWithValidHits >= 2'),
    debug = cms.untracked.bool(False),
    errorRescaleFactor = cms.double(2.0),
    fromVertex = cms.bool(False),
    hitCollector = cms.string('hitCollectorForOutInMuonSeeds'),
    hitsToTry = cms.int32(3),
    layersToTry = cms.int32(3),
    maxEtaForTOB = cms.double(1.8),
    minEtaForTEC = cms.double(0.7),
    muonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    src = cms.InputTag("earlyDisplacedMuons"),
    trackerPropagator = cms.string('PropagatorWithMaterial')
)
