import FWCore.ParameterSet.Config as cms

MEtoMEComparitor = cms.EDAnalyzer("MEtoMEComparitor",
                                  MEtoEDMLabel = cms.string('MEtoEDMConverter'),
                                  
                                  lumiInstance = cms.string('MEtoEDMConverterLumi'),
                                  runInstance = cms.string('MEtoEDMConverterRun'),

                                  autoProcess = cms.bool(False),
                                  processRef = cms.string('HLT'),
                                  processNew = cms.string('RERECO'),

                                  #under which an histogram goes to diff check
                                  KSgoodness = cms.double(0.9),
                                  #over which an histogram is marked as badKS, badDiff
                                  Diffgoodness = cms.double(0.1),
                                  #deepness of directory
                                  dirDepth = cms.uint32(1),
                                  #fraction in a directory under which a warning is send
                                  OverAllgoodness = cms.double(0.9),
                                  
                                  )
