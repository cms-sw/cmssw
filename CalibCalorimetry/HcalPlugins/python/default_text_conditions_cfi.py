import FWCore.ParameterSet.Config as cms

# HCAL setup suitable for MC simulation and production (no ElectronicsMapping)
hcal_db_producer = cms.ESProducer("HcalDbProducer")

es_ascii = cms.ESSource("HcalTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('Pedestals'),
        file = cms.FileInPath('CondFormats/HcalObjects/data/pedestals_hardcoded.txt')
    ), 
        cms.PSet(
            object = cms.string('PedestalWidths'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/pedestal_widths_hardcoded.txt')
        ), 
        cms.PSet(
            object = cms.string('Gains'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/gains_hardcoded.txt')
        ), 
        cms.PSet(
            object = cms.string('GainWidths'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/gain_widths_hardcoded.txt')
        ), 
        cms.PSet(
            object = cms.string('QIEData'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/qie_hardcoded.txt')
        ), 
        cms.PSet(
            object = cms.string('ElectronicsMap'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/emap_tb2006_v7.txt')
        ), 
        cms.PSet(
            object = cms.string('ChannelQuality'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/quality_hardcoded.txt')
        ))
)


