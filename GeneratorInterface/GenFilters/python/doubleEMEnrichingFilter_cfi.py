import FWCore.ParameterSet.Config as cms

doubleEMenrichingfilter = cms.EDFilter("doubleEMEnrichingFilter",
    filterAlgoPSet = cms.PSet(
        requireTrackMatch = cms.bool(False),
        caloIsoMax = cms.double(3.0),
        isoGenParConeSize = cms.double(0.1),
        tkIsoMax = cms.double(3.0),
        isoConeSize = cms.double(0.2),
        isoGenParETMin = cms.double(4.0),
        hOverEMax = cms.double(0.5),
        clusterThreshold = cms.double(4.0),
        seedThreshold = cms.double(2.5),
        eTThreshold = cms.double(3.0)
    )
)


