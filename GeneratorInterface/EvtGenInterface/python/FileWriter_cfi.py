import FWCore.ParameterSet.Config as cms

DataCardFileWriter = cms.EDAnalyzer("DataCardFileWriter",
                                    FileName = cms.string("MyDEC.DEC"),
                                    FileContent = cms.vstring("Hello World","Bye World")
                                    )
