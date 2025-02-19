import FWCore.ParameterSet.Config as cms

emenrichingfilter = cms.EDFilter("EMEnrichingFilter",
    filterAlgoPSet = cms.PSet(
        requireTrackMatch = cms.bool(False),
        caloIsoMax = cms.double(10.0),
        isoGenParConeSize = cms.double(0.1),
        tkIsoMax = cms.double(5.0),
        isoConeSize = cms.double(0.2),
        isoGenParETMin = cms.double(20.0),
        hOverEMax = cms.double(0.5),
        clusterThreshold = cms.double(20.0)
    )
)


