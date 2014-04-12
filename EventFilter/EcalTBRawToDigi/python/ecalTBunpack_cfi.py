import FWCore.ParameterSet.Config as cms

ecalTBunpack = cms.EDProducer("EcalDCCTBUnpackingModule",
                              fedRawDataCollectionTag = cms.InputTag('rawDataCollector')
)
