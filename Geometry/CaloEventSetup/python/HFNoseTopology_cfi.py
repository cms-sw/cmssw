import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal Topologies
#
HFNoseTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                       Name = cms.string("HGCalHFNoseSensitive"),
                                       Type = cms.int32(6)
                                       )

