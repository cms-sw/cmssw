import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

siStripZeroSuppression = cms.EDProducer("SiStripZeroSuppression",
                                        Algorithms = DefaultAlgorithms,
                                        RawDigiProducersList = cms.VInputTag( cms.InputTag('siStripDigis','VirginRaw'), 
                                                                              cms.InputTag('siStripDigis','ProcessedRaw'),
                                                                              cms.InputTag('siStripDigis','ScopeMode')),
                                        storeCM = cms.bool(False), 
                                        produceRawDigis = cms.bool(False), # if mergeCollection is True, produceRawDigi is not considered
                                        mergeCollections = cms.bool(False)
                                        )
