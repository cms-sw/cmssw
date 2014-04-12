import FWCore.ParameterSet.Config as cms

ecalEBunpacker = cms.EDProducer("EcalDCCTBUnpackingModule",
                                fedRawDataCollectionTag = cms.InputTag('rawDataCollector')
)
