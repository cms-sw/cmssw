import FWCore.ParameterSet.Config as cms

SiStripMeanCMExtractor = cms.EDProducer("SiStripMeanCMExtractor",

    CMCollection = cms.InputTag('siStripZeroSuppression','APVCM'), 
    Algorithm = cms.string("Pedestals"),
    NEvents = cms.uint32(100)
	
)
