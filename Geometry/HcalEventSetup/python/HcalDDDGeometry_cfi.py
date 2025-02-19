import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the HCAL geometry model using DDD
#
HcalDDDGeometryEP = cms.ESProducer("HcalDDDGeometryEP")

CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL')
)


