import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

siStripZeroSuppression = cms.EDFilter("SiStripZeroSuppression",
                                      Algorithms = DefaultAlgorithms,
                                      RawDigiProducersList = cms.VInputTag( cms.InputTag('simSiStripDigis','VirginRaw'), 
                                                                            cms.InputTag('simSiStripDigis','ProcessedRaw'),
                                                                            cms.InputTag('simSiStripDigis','ScopeMode'))
                                      )



