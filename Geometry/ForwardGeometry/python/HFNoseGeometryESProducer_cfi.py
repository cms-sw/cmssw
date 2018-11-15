import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HFNose Geometry
#

HFNoseGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
                                          Name = cms.untracked.string("HGCalHFNoseSensitive")
                                          )
