import FWCore.ParameterSet.Config as cms

ecalEBunpacker = cms.EDProducer("EcalDCCTBUnpackingModule",
                                fedRawDataCollectionTag = cms.InputTag('rawDataCollector')
)
# foo bar baz
# q2kgcwwoE63zf
