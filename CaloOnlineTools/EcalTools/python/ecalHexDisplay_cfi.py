import FWCore.ParameterSet.Config as cms

hexDump = cms.EDAnalyzer("EcalHexDisplay",
    fedRawDataCollectionTag = cms.InputTag('rawDataCollector'),
    verbosity = cms.untracked.int32(0),
    filename = cms.untracked.string('dump.bin'),
    # fed_id: EE- is 601-609,  EB is 610-645,  EE- is 646-654
    # when using 'single sm' fed corresponds to construction number  
    beg_fed_id = cms.untracked.int32(0),
    writeDCC = cms.untracked.bool(False),
    end_fed_id = cms.untracked.int32(654)
)
