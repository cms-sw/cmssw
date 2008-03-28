import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the Forward geometry model
#
ZdcHardcodeGeometryEP = cms.ESProducer("ZdcHardcodeGeometryEP")

CastorHardcodeGeometryEP = cms.ESProducer("CastorHardcodeGeometryEP")


