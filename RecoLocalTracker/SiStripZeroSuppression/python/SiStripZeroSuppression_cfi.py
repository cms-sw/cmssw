import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms import *

siStripZeroSuppression = cms.EDFilter("SiStripZeroSuppression",
                                      Algorithms = DefaultAlgorithms,
                                      RawDigiProducersList = cms.VInputTag( cms.InputTag('siStripDigis','VirginRaw'), 
                                                                            cms.InputTag('siStripDigis','ProcessedRaw'),
                                                                            cms.InputTag('siStripDigis','ScopeMode'))
                                      )
