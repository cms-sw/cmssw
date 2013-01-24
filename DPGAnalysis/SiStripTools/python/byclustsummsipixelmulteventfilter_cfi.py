import FWCore.ParameterSet.Config as cms

byclustsummsipixelmulteventfilter = cms.EDFilter('ByClusterSummarySingleMultiplicityEventFilter',
                                                 multiplicityConfig = cms.PSet(
                                                                 clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                                 subDetEnum = cms.int32(5),
                                                                 subDetVariable = cms.string("pHits")
                                                                 ),
                                                 cut = cms.string("mult > 300")
                                                 )
	
