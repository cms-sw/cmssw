import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal Topologies
#
HGCalEETopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                        Name = cms.untracked.string("HGCalEESensitive"),
                                        Type = cms.untracked.int32(2)
                                        )


HGCalHESilTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                           Name = cms.untracked.string("HGCalHESiliconSensitive"),
                                           Type = cms.untracked.int32(3)
                                           )


HGCalHESciTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                           Name = cms.untracked.string("HGCalHEScintillatorSensitive"),
                                           Type = cms.untracked.int32(4)
                                           )

