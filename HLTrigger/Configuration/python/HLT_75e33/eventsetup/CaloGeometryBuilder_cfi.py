import FWCore.ParameterSet.Config as cms

CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring(
        'HCAL',
        'ZDC',
        'EcalBarrel',
        'TOWER',
        'HGCalEESensitive',
        'HGCalHESiliconSensitive',
        'HGCalHEScintillatorSensitive'
    )
)
