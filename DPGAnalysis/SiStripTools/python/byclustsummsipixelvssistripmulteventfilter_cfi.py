import FWCore.ParameterSet.Config as cms

byclustsummsipixelvssistripmulteventfilter = cms.EDFilter('ByClusterSummaryMultiplicityPairEventFilter',
                                                          multiplicityConfig = cms.PSet(
                                                                           firstMultiplicityConfig = cms.PSet(
                                                                                                     clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                                                                     subDetEnum = cms.int32(5),
                                                                                                     subDetVariable = cms.string("pHits")
                                                                                                     ),
                                                                           secondMultiplicityConfig = cms.PSet(
                                                                                                      clusterSummaryCollection = cms.InputTag("clusterSummaryProducer"),
                                                                                                      subDetEnum = cms.int32(0),
                                                                                                      subDetVariable = cms.string("cHits")
                                                                                                      ),
                                                                           ),
                                                          cut = cms.string("(mult2 > 30000) && ( mult2 > 20000+7*mult1)")
                                                          )
	
