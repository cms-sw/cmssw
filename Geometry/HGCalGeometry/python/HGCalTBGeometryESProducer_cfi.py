import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HGCal TB Geometry
#

HGCalTBEEGeometryESProducer = cms.ESProducer("HGCalTBGeometryESProducer",
                                             Name = cms.untracked.string("HGCalEESensitive") )


HGCalTBHESilGeometryESProducer = cms.ESProducer("HGCalTBGeometryESProducer",
                                                Name = cms.untracked.string("HGCalHESiliconSensitive") )

