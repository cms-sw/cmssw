import FWCore.ParameterSet.Config as cms

# HCAL setup suitable for processing TB04 data using hardwired calibrations
hcal_db_producer = cms.ESProducer("HcalDbProducer")

hcal_es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('Pedestals', 
        'PedestalWidths', 
        'Gains', 
        'GainWidths', 
        'QIEShape', 
        'QIEData', 
        'channelQuality')
)

hcal_es_ascii = cms.ESSource("HcalTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('ElectronicsMap'),
        file = cms.FileInPath('CondFormats/HcalObjects/data/hbho_ring12_tb04.txt')
    ))
)


