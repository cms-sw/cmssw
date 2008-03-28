import FWCore.ParameterSet.Config as cms

# HCAL setup suitable for MC simulation and production (no ElectronicsMapping)
hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_ascii = cms.ESSource("HcalTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('Pedestals'),
        file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_pedestals_fC_v1_zdc.txt')
    ), cms.PSet(
        object = cms.string('PedestalWidths'),
        file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_widths_fC_v1_zdc.txt')
    ), cms.PSet(
        object = cms.string('Gains'),
        file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_gains_v1_zdc.txt')
    ), cms.PSet(
        object = cms.string('QIEData'),
        file = cms.FileInPath('CondFormats/HcalObjects/data/qie_normalmode_v3_zdc.txt')
    ), cms.PSet(
        object = cms.string('ElectronicsMap'),
        file = cms.FileInPath('CondFormats/HcalObjects/data/official_emap_16x_v4.txt')
    ))
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 'channelQuality')
)


