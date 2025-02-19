import FWCore.ParameterSet.Config as cms

# HCAL setup suitable for MC simulation and production
hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_ascii = cms.ESSource("HcalTextCalibrations",
    input = cms.VPSet(
        cms.PSet(
            object = cms.string('Pedestals'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_pedestals_fC_v6_mc.txt')
        ), 
        cms.PSet(
            object = cms.string('PedestalWidths'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_widths_fC_v6_mc.txt')
        ), 
        cms.PSet(
            object = cms.string('Gains'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/gains-mc/HBflatmax8.5kV_brightened_HEflatmean7750Vmax8000V_HO8kV_real_HF1250V_withcorrections_080827.txt')
        ), 
        cms.PSet(
            object = cms.string('QIEData'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/qie_normalmode_v6_cand2_fakeZDC.txt')
        ), 
        cms.PSet(
            object = cms.string('GainWidths'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_gains_widths_v1.txt')
        ), 
        cms.PSet(
            object = cms.string('ElectronicsMap'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/official_emap_v7.00_081109.txt')
        ), 
        cms.PSet(
            object = cms.string('ChannelQuality'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_channelStatus_default.txt')
        ), 
        cms.PSet(
            object = cms.string('RespCorrs'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_respCorr_trivial_HF0.7.txt')
        ), 
        cms.PSet(
            object = cms.string('TimeCorrs'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_timeCorr_mc_v1.txt')
        ), 
        cms.PSet(
            object = cms.string('ZSThresholds'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_ZSthresholds_default.txt')
        ),
        cms.PSet(
            object = cms.string('L1TriggerObjects'),
            file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_L1TriggerObject_trivial.txt')
        )
     )
)


