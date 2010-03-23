import FWCore.ParameterSet.Config as cms

HiTrivialConditionRetriever = cms.ESSource('HiTrivialConditionRetriever',
                                           inputFile = cms.string("RecoHI/HiCentralityAlgos/data/CentralityTablesHydjet2760GeV.txt"),
                                           verbose = cms.untracked.int32(1)
                                           )
