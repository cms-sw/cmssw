import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal Topologies
#
HGCalEETopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                        Name = cms.untracked.string("HGCalEESensitive"),
                                        Type = cms.untracked.int32(0)
                                        )


HGCalHESilTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                           Name = cms.untracked.string("HGCalHESiliconSensitive"),
                                           Type = cms.untracked.int32(1)
                                           )


HGCalHESciTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                           Name = cms.untracked.string("HGCalHEScintillatorSensitive"),
                                           Type = cms.untracked.int32(2)
                                           )

