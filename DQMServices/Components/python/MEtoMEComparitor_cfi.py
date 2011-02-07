import FWCore.ParameterSet.Config as cms

MEtoMEComparitor = cms.EDAnalyzer("MEtoMEComparitor",
                                  MEtoEDMLabel = cms.string('MEtoEDMConverter'),
                                  
                                  lumiInstance = cms.string('MEtoEDMConverterLumi'),
                                  runInstance = cms.string('MEtoEDMConverterRun'),

                                  autoProcess = cms.bool(False),
                                  processRef = cms.string('HLT'),
                                  processNew = cms.string('RERECO'),
                                  
                                  KSgoodness = cms.double(0.9)
                                  )
