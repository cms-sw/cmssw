import FWCore.ParameterSet.Config as cms

cosmicsNavigationSchoolESProducer = cms.ESProducer("SkippingLayerCosmicNavigationSchoolESProducer",
    ComponentName = cms.string('CosmicNavigationSchool'),
    allSelf = cms.bool(True),
    noPXB = cms.bool(False),
    noPXF = cms.bool(False),
    noTEC = cms.bool(False),
    noTIB = cms.bool(False),
    noTID = cms.bool(False),
    noTOB = cms.bool(False),
    selfSearch = cms.bool(True)
)
