import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the ECAL + HCAL geometry model
#
EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
                                      applyAlignment = cms.untracked.bool(false) )

EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP",
                                      applyAlignment = cms.untracked.bool(false))

EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP",
                                      applyAlignment = cms.untracked.bool(false))

HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP",
                                      applyAlignment = cms.untracked.bool(false))

ZdcHardcodeGeometryEP = cms.ESProducer("ZdcHardcodeGeometryEP",
                                      applyAlignment = cms.untracked.bool(false))

CaloTowerHardcodeGeometryEP = cms.ESProducer("CaloTowerHardcodeGeometryEP")

CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL',
                                'ZDC',
                                'EcalBarrel',
                                'EcalEndcap',
                                'EcalPreshower',
                                'TOWER')
)


