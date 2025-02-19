import FWCore.ParameterSet.Config as cms

process = cms.Process("SM")

process.source = cms.Source("FragmentInput")

process.out1 = cms.OutputModule("EventStreamFileWriter",
                                streamLabel = cms.string('A'),
                                maxSize = cms.int32(1024),
                                SelectHLTOutput = cms.untracked.string('hltOutputDQM'),
                                fractionToDisk = cms.untracked.double( 1.0 )
                                )

process.out4 = cms.OutputModule("ErrorStreamFileWriter",
                                streamLabel = cms.string('Error'),
                                maxSize = cms.int32(1)
                                )

process.end1 = cms.EndPath(process.out1)
process.end4 = cms.EndPath(process.out4)
