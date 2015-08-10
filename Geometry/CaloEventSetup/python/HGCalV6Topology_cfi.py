import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal Topologies
#
HGCalEETopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                        Name = cms.untracked.string("HGCalEESensitive"),
                                        Type = cms.untracked.int32(0),
                                        HalfType = cms.untracked.bool(False)
                                        )


HGCalHESilTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
                                           Name = cms.untracked.string("HGCalHESiliconSensitive"),
                                           Type = cms.untracked.int32(1),
                                           HalfType = cms.untracked.bool(False)
                                           )

