import FWCore.ParameterSet.Config as cms

ecalTBunpack = cms.EDProducer("EcalDCCTBUnpackingModule",
                              fedRawDataCollectionTag = cms.InputTag('rawDataCollector')
)
# foo bar baz
# 6UpGDNQPMb9Z2
# FVlLl2IrkBTGy
