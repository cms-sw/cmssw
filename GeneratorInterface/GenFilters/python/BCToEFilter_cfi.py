import FWCore.ParameterSet.Config as cms

bctoefilter = cms.EDFilter("BCToEFilter",
    filterAlgoPSet = cms.PSet(
        maxAbsEta = cms.double(3.05),
        eTThreshold = cms.double(10.0)
    )
)


