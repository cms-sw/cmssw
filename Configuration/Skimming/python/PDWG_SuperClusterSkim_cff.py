import FWCore.ParameterSet.Config as cms

### Diphoton CS                                                                                                                       
superClusterMerger =  cms.EDProducer("EgammaSuperClusterMerger",
                                   src = cms.VInputTag(cms.InputTag('correctedHybridSuperClusters'),
                                                       cms.InputTag('correctedMulti5x5SuperClustersWithPreshower'))
                                   )
superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
                                   src = cms.InputTag("superClusterMerger"),
                                   particleType = cms.string('gamma')
                                   )
superClusterCandsPt20 = cms.EDFilter("CandPtrSelector",  
                                     src = cms.InputTag("superClusterCands"),
                                     cut  = cms.string("pt > 20"),
                                     )
#superClusterCandsPt50 = cms.EDFilter("CandPtrSelector",  
#                                     src = cms.InputTag("superClusterCands"),
#                                     cut  = cms.string("pt > 50"),
#                                     filter = cms.bool(True),
#                                     )
twoSuperClustersPt20 = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("superClusterCandsPt20"),
                                    minNumber = cms.uint32(2),
                                    )


diSuperClusterSkimSequence = cms.Sequence(
    superClusterMerger *
    superClusterCands *
    superClusterCandsPt20 *
    twoSuperClustersPt20
    )

#singleSuperClusterSkimSequence = cms.Sequence(
#    superClusterMerger *
#    superClusterCands *
#    superClusterCandsPt50
#    )

