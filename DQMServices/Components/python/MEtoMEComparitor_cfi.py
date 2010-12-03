import FWCore.ParameterSet.Config as cms

MEtoMEComparitor = cms.EDAnalyzer("MEtoMEComparitor",
                                  MEtoEDMTag_ref = cms.InputTag('MEtoEDMConverter','','HLT'),
                                  MEtoEDMTag_new = cms.InputTag('MEtoEDMConverter','','RERECO'),
                                  lumiInstance = cms.string('MEtoEDMConverterLumi'),
                                  runInstance = cms.string('MEtoEDMConverterRun'),
                                  
                                  KSgoodness = cms.double(0.9)
                                  )
