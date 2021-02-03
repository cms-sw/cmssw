import FWCore.ParameterSet.Config as cms

RegionPSetBlock = cms.PSet(
    RegionPSet = cms.PSet(
        originHalfLength = cms.double(21.2),
        originRadius = cms.double(0.2),
        originXPos = cms.double(0.0),
        originYPos = cms.double(0.0),
        originZPos = cms.double(0.0),
        precise = cms.bool(True),
        ptMin = cms.double(0.9),
        useMultipleScattering = cms.bool(False)
    )
)