import FWCore.ParameterSet.Config as cms

#defaults as in SiStripZeroSuppression_cfi.py
siStripProcessedRawDigis = cms.EDProducer("SiStripProcessedRawDigiProducer",
                                          DigiProducersList = cms.VInputTag( cms.InputTag('siStripDigis','ZeroSuppressed'),
                                                                             cms.InputTag('siStripDigis','VirginRaw'), 
                                                                             cms.InputTag('siStripDigis','ProcessedRaw'),
                                                                             cms.InputTag('siStripDigis','ScopeMode')
                                                                             ),
                                          CommonModeNoiseSubtractionMode = cms.string('Median'), 
                                          #CutToAvoidSignal = cms.double(3.0), ##This is just for CMNSub...Mode TT6
                                          )
