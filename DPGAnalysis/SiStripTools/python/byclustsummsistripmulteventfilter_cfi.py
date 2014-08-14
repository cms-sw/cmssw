import FWCore.ParameterSet.Config as cms

byclustsummsistripmulteventfilter = cms.EDFilter('ByClusterSummarySingleMultiplicityEventFilter',
                                                 multiplicityConfig = cms.PSet(
                                                                 clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                                 subDetEnum = cms.int32(0),
                                                                 varEnum = cms.int32(0)
                                                                 ),
                                                 cut = cms.string("mult > 300")
                                                 )
	
