import FWCore.ParameterSet.Config as cms

process = cms.Process("SM")

process.source = cms.Source("FragmentInput")

process.out4 = cms.OutputModule("ErrorStreamFileWriter",
                                streamLabel = cms.string('Error'),
                                maxSize = cms.int32(1)
                                )

process.end4 = cms.EndPath(process.out4)
