import FWCore.ParameterSet.Config as cms

ecalPreshowerDigis = cms.EDProducer("ESRawToDigi",
    ESdigiCollection = cms.string(''),
    InstanceES = cms.string(''),
    LookupTable = cms.FileInPath('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat'),
    debugMode = cms.untracked.bool(False),
    mightGet = cms.optional.untracked.vstring,
    sourceTag = cms.InputTag("rawDataCollector")
)
