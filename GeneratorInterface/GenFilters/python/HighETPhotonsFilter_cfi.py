import FWCore.ParameterSet.Config as cms

highetphotonsfilter = cms.EDFilter("HighETPhotonsFilter",
    filterAlgoPSet = cms.PSet(
        sumETThreshold = cms.double(90.0),
        seedETThreshold = cms.double(40.0),
        nonPhotETMax = cms.double(10.0),
        isoConeSize = cms.double(0.1)
    )
)


