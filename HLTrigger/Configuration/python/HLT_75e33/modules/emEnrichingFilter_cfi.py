import FWCore.ParameterSet.Config as cms

emEnrichingFilter = cms.EDFilter("EMEnrichingFilter",
    filterAlgoPSet = cms.PSet(
        caloIsoMax = cms.double(10.0),
        clusterThreshold = cms.double(20.0),
        genParSource = cms.InputTag("genParticles"),
        hOverEMax = cms.double(0.5),
        isoConeSize = cms.double(0.2),
        isoGenParConeSize = cms.double(0.1),
        isoGenParETMin = cms.double(20.0),
        requireTrackMatch = cms.bool(False),
        tkIsoMax = cms.double(5.0)
    )
)
