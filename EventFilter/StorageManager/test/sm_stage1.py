import FWCore.ParameterSet.Config as cms

process = cms.Process("SM")

process.source = cms.Source("FragmentInput")

process.out1 = cms.OutputModule("EventStreamFileWriter",
                                streamLabel = cms.string('A'),
                                maxSize = cms.int32(512),
                                SelectHLTOutput = cms.untracked.string('out4DQM')
                                )

process.out2 = cms.OutputModule("EventStreamFileWriter",
                                streamLabel = cms.string('B'),
                                maxSize = cms.int32(1024),
                                SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('*') ),
                                SelectHLTOutput = cms.untracked.string('PhysicsOModule')
                                )

process.out3 = cms.OutputModule("EventStreamFileWriter",
                                streamLabel = cms.string('C'),
                                maxSize = cms.int32(200),
                                SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('DiMuon', 'HighPT') ),
                                SelectHLTOutput = cms.untracked.string('PhysicsOModule')
                                )

process.out4 = cms.OutputModule("ErrorStreamFileWriter",
                                streamLabel = cms.string('Error'),
                                maxSize = cms.int32(32)
                                )

process.end1 = cms.EndPath(process.out1)
process.end2 = cms.EndPath(process.out2)
process.end3 = cms.EndPath(process.out3)
process.end4 = cms.EndPath(process.out4)
