import FWCore.ParameterSet.Config as cms

hcal_db_producer = cms.ESProducer("HcalDbProducer",
     dump = cms.untracked.vstring(''),
     file = cms.untracked.string('')
)
 
hcales_ascii = cms.ESSource("HcalTextCalibrations",
                        input = cms.VPSet(
    cms.PSet(
    object = cms.string('ElectronicsMap'),
    file = cms.FileInPath('RecoTBCalo/HcalPlotter/data/tb2010_map.txt')
    ),
    cms.PSet(
    object = cms.string('Pedestals'),
    file = cms.FileInPath('RecoTBCalo/HcalPlotter/data/ped_tb2010_000495.txt')
    )
    )
                                )

hcales_hardcode = cms.ESSource("HcalHardcodeCalibrations",
                               toGet = cms.untracked.vstring('PedestalWidths', 'LutMetadata',
                                                             'Gains', 'GainWidths', 'LUTCorrs', 
                                                             'PFCorrs', 'QIEData',
                                                             'L1TriggerObjects','ZSThresholds','DcsValues',
                                                             'ChannelQuality','RespCorrs','TimeCorrs')
                               ) 
