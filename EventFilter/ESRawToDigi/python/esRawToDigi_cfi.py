import FWCore.ParameterSet.Config as cms

esRawToDigi = cms.EDFilter("ESRawToDigi",
    debugMode = cms.untracked.bool(False),
    InstanceES = cms.string(''),
    ESdigiCollection = cms.string(''),
    Label = cms.string('source')
)


