import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal TB Topologies
#
HGCalTBEETopologyBuilder = cms.ESProducer("HGCalTBTopologyBuilder",
                                          Name = cms.string("HGCalEESensitive"),
                                          Type = cms.int32(3) )


HGCalTBHESilTopologyBuilder = cms.ESProducer("HGCalTBTopologyBuilder",
                                             Name = cms.string("HGCalHESiliconSensitive"),
                                             Type = cms.int32(4) )

