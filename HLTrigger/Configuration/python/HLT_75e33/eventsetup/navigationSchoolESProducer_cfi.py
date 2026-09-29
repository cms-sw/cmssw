import FWCore.ParameterSet.Config as cms

navigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('SimpleNavigationSchool'),
    PluginName = cms.string('SimpleNavigationSchool'),
    SimpleMagneticField = cms.string(''),
    appendToDataLabel = cms.string('')
)
