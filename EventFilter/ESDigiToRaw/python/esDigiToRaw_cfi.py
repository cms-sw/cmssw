import FWCore.ParameterSet.Config as cms

esDigiToRaw = cms.EDProducer("ESDigiToRaw",
    debugMode = cms.untracked.bool(False),
    InstanceES = cms.string(''),
    Label = cms.string('simEcalPreshowerDigis'),
    LookupTable = cms.untracked.FileInPath('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat')
)
