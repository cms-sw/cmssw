import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.source = cms.Source("FragmentInput")

#process.out3  = cms.OutputModule("EventStreamFileWriter",
#                                 streamLabel = cms.string('C'),
#                                 maxSize = cms.int32(1073741824),
#				 SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p0') )
#                                )      

#process.e3 = cms.EndPath(process.out3)

