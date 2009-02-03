import FWCore.ParameterSet.Config as cms

#defaults as in SiStripZeroSuppression_cfi.py
siStripProcessedRawDigis = cms.EDProducer("SiStripProcessedRawDigiProducer",
                                          CommonModeNoiseSubtractionMode = cms.string('Median'), 
                                          CutToAvoidSignal = cms.double(3.0), ##This is just for CMNSub...Mode TT6
                                          DigiProducersList = cms.VPSet( cms.PSet(DigiProducer = cms.InputTag('siStripDigis', 'ZeroSuppressed') ),
                                                                         cms.PSet(DigiProducer = cms.InputTag('siStripDigis', 'VirginRaw'     ) ),
                                                                         cms.PSet(DigiProducer = cms.InputTag('siStripDigis', 'ProcessedRaw'  ) ),
                                                                         cms.PSet(DigiProducer = cms.InputTag('siStripDigis', 'ScopeMode'     ) )
                                                                         )
                                          )
