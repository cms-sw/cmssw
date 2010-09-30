import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

siStripZeroSuppression = cms.EDProducer("SiStripZeroSuppression",
                                        Algorithms = DefaultAlgorithms,
                                        RawDigiProducersList = cms.VInputTag( cms.InputTag('simSiStripDigis','VirginRaw'), 
                                                                            cms.InputTag('simSiStripDigis','ProcessedRaw'),
                                                                            cms.InputTag('simSiStripDigis','ScopeMode')),
                                        storeCM = cms.bool(True), 
                                        produceRawDigis = cms.bool(True), # if mergeCollection is True, produceRawDigi is not considered
                                        mergeCollections = cms.bool(False)
                                        )



