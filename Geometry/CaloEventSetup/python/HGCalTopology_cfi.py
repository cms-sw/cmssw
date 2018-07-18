import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal Topologies
#
HGCalEETopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                        Name = cms.string("HGCalEESensitive"),
                                        Type = cms.int32(3)
                                        )


HGCalHESilTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                           Name = cms.string("HGCalHESiliconSensitive"),
                                           Type = cms.int32(4)
                                           )


HGCalHESciTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                           Name = cms.string("HGCalHEScintillatorSensitive"),
                                           Type = cms.untracked.int32(5)
                                           )

