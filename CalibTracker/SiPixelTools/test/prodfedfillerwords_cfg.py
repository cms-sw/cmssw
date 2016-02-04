import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.source = cms.Source("PoolSource",                         
                            fileNames = cms.untracked.vstring ('file:/afs/cern.ch/user/f/florez/CMSSW_2_0_4/src/FedFillerWords/Data/mysimple.root') 
                            )
			    
process.MessageLogger = cms.Service("MessageLogger", 
                                    destinations = cms.untracked.vstring('output'),
                                    threshold = cms.untracked.string('INFO'),                              
                                    noLineBreaks = cms.untracked.bool(True)
                                    )
				    
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('output'),
				    threshold = cms.untracked.string('ERROR'),
				    noLineBreaks = cms.untracked.bool(True)
				    )
				    
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('output'),
				    threshold = cms.untracked.string('WARNING'),
				    noLineBreaks = cms.untracked.bool(True)
				    )
				    				    
process.filler = cms.EDProducer("SiPixelFedFillerWordEventNumber",
                                InputLabel = cms.untracked.string('source'),
                                InputInstance = cms.untracked.string(''),
                                SaveFillerWords = cms.bool(False)
                                )

process.out = cms.EDProducer("PoolOutputModule", 
                             fileName = cms.untracked.string("NewFEDFWs.root")
                             )
			     
process.producer = cms.Path(process.filler) 
process.producer = cms.EndPath(process.out) 
