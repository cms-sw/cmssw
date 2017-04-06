import FWCore.ParameterSet.Config as cms

byclustsummsipixelmulteventfilter = cms.EDFilter('ByClusterSummarySingleMultiplicityEventFilter',
                                                 multiplicityConfig = cms.PSet(
                                                                 clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                                 subDetEnum = cms.int32(5),
                                                                 varEnum = cms.int32(0)
                                                                 ),
                                                 cut = cms.string("mult > 300")
                                                 )
	
