import FWCore.ParameterSet.Config as cms

byclustsummsistripmulteventfilter = cms.EDFilter('ByClusterSummarySingleMultiplicityEventFilter',
                                                 multiplicityConfig = cms.PSet(
                                                                 clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                                 subDetEnum = cms.int32(0),
                                                                 subDetVariable = cms.string("cHits")
                                                                 ),
                                                 cut = cms.string("mult > 300")
                                                 )
	
