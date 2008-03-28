import FWCore.ParameterSet.Config as cms

esDigiToRaw = cms.EDFilter("ESDigiToRaw",
    debugMode = cms.untracked.bool(False),
    InstanceES = cms.string(''),
    Label = cms.string('ecalPreshowerDigis')
)


