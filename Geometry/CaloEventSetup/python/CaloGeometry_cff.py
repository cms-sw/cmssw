import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the ECAL + HCAL geometry model
#
EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP")

EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP")

EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP")

HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP")

ZdcHardcodeGeometryEP = cms.ESProducer("ZdcHardcodeGeometryEP")

CaloTowerHardcodeGeometryEP = cms.ESProducer("CaloTowerHardcodeGeometryEP")

CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL', 
        'ZDC', 
        'EcalBarrel', 
        'EcalEndcap', 
        'EcalPreshower', 
        'TOWER')
)


