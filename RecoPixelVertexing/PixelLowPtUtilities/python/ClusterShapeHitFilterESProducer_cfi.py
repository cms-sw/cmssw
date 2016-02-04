import FWCore.ParameterSet.Config as cms

ClusterShapeHitFilterESProducer = cms.ESProducer("ClusterShapeHitFilterESProducer",
                                                        ComponentName = cms.string('ClusterShapeHitFilter')
                                                        )
