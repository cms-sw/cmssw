import FWCore.ParameterSet.Config as cms

bctoefilter = cms.EDFilter("BCToEFilter",
    filterAlgoPSet = cms.PSet(
        eTThreshold = cms.double(10.0)
    )
)


