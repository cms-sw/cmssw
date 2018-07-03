import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the FastTime Topologies
#
FastTimeBarrelTopologyBuilder = cms.ESProducer("FastTimeTopologyBuilder",
                                               Name = cms.untracked.string("FastTimeBarrel"),
                                               Type = cms.untracked.int32(1)
                                               )


FastTimeEndcapTopologyBuilder = cms.ESProducer("FastTimeTopologyBuilder",
                                               Name = cms.untracked.string("SFBX"),
                                               Type = cms.untracked.int32(2)
                                               )

