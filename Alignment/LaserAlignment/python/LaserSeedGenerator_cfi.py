import FWCore.ParameterSet.Config as cms

LaserSeedFinder = cms.EDFilter("LaserSeedGenerator",
    originHalfLength = cms.double(90.0),
    originZPosition = cms.double(0.0),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    TTRHBuilder = cms.string('WithTrackAngle'),
    ptMin = cms.double(0.0),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    originRadius = cms.double(150.0),
    Propagator = cms.string('Analytical'), ## change to WithMaterial if you want to use the PropagatorWithMaterial

    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit")
)


