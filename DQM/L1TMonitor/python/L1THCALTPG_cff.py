import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1THCALTPG_cfi import *
hcalunpacker = cms.EDFilter("HcalRawToDigi",
    # Optional filter to remove any digi with "data valid" off, "error" on, 
    # or capids not rotating
    FilterDataQuality = cms.bool(True),
    # Do not complain about missing FEDs
    ExceptionEmptyData = cms.untracked.bool(False),
    HcalFirstFED = cms.untracked.int32(700),
    InputLabel = cms.InputTag("source"),
    # Use the defaults for FED numbers
    # Do not complain about missing FEDs
    ComplainEmptyData = cms.untracked.bool(False),
    # Flag to enable unpacking of calibration channels (default = false)
    UnpackCalib = cms.untracked.bool(False),
    FEDs = cms.untracked.vint32(719, 721, 723),
    lastSample = cms.int32(9),
    # At most ten samples can be put into a digi, if there are more
    # than ten, firstSample and lastSample select which samples
    # will be copied to the digi
    firstSample = cms.int32(0)
)

HcalDbProducer = cms.ESProducer("HcalDbProducer")

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('Gains', 
        'GainWidths', 
        'QIEShape', 
        'QIEData', 
        'ChannelQuality')
)

es_ascii = cms.ESSource("HcalTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('Pedestals'),
        file = cms.FileInPath('DQM/HcalMonitorModule/test/peds_ADC_hfplus_000269.txt')
    ), 
        cms.PSet(
            object = cms.string('PedestalWidths'),
            file = cms.FileInPath('DQM/HcalMonitorModule/test/widths_ADC_hfplus_000269.txt')
        ), 
        cms.PSet(
            object = cms.string('ElectronicsMap'),
            file = cms.FileInPath('DQM/HcalMonitorModule/data/test_tp_map.txt')
        ))
)

l1thcaltpgpath = cms.Path(hcalunpacker*l1thcaltpg)

