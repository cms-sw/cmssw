import FWCore.ParameterSet.Config as cms

esRawToDigi = cms.EDProducer("ESRawToDigi",
    debugMode = cms.untracked.bool(False),
    InstanceES = cms.string(''),
    ESdigiCollection = cms.string(''),
    sourceTag = cms.InputTag('rawDataCollector'),
    LookupTable = cms.FileInPath('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat')
)


