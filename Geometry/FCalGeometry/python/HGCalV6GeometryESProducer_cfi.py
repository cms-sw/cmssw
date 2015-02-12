import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal Geometry
#

HGCalEEGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
                                           Name = cms.untracked.string("HGCalEESensitive")
                                           )


HGCalHESilGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
                                              Name = cms.untracked.string("HGCalHESiliconSensitive")
                                              )

