import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal Topologies
#
HGCalEETopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                        Name = cms.string("HGCalEESensitive"),
                                        Type = cms.int32(8)
                                        )
