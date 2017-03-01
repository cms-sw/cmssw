import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the FastTime Geometry
#

FastTimeBarrelGeometryESProducer = cms.ESProducer("FastTimeGeometryESProducer",
                                                  Name = cms.untracked.string("FastTimeBarrel"),
                                                  Type = cms.untracked.int32(1)
                                                  )


FastTimeEndcapGeometryESProducer = cms.ESProducer("FastTimeGeometryESProducer",
                                                  Name = cms.untracked.string("SFBX"),
                                                  Type = cms.untracked.int32(2)
                                                  )
