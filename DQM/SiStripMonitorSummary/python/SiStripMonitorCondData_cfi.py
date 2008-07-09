# The following comments couldn't be translated into the new config version:

# combinations of "TIB","TOB","TID","TEC" 
# *** other restricting specifications

import FWCore.ParameterSet.Config as cms

CondDataMonitoring = cms.EDFilter("SiStripMonitorCondData",
    OutputMEsInRootFile = cms.bool(True),
    # -----
    SiStripNoisesDQM_PSet = cms.PSet(
        SummaryOfCumul_NchX = cms.int32(11),
        SummaryOfCumul_xTitle = cms.string('Noise from CondDB(ADC)'),
        Profile_xTitle = cms.string('Strip Number'),
        CondObj_name = cms.string('noise'),
        CondObj_fillId = cms.string('onlyProfile'),
        Cumul_xTitle = cms.string('Noise from CondDB(ADC)'),
        Cumul_description = cms.string('NoiseFromCondDB_Cumul'),
        SummaryOfCumul_HighX = cms.double(10.5),
        SummaryOfProfile_yTitle = cms.string('Noise from CondDB(ADC)'),
        Cumul_NchX = cms.int32(11),
        SummaryOfCumul_yTitle = cms.string(' '),
        SummaryOfProfile_description = cms.string('NoiseFromCondDB_ProfileSummary'),
        Cumul_yTitle = cms.string(' '),
        SummaryOfCumul_description = cms.string('NoiseFromCondDB_CumulSummary'),
        Profile_description = cms.string('NoiseFromCondDB_Profile'),
        SummaryOfProfile_xTitle = cms.string('Strip Number'),
        Cumul_LowX = cms.double(-0.5),
        SummaryOfCumul_LowX = cms.double(-0.5),
        Cumul_HighX = cms.double(10.5),
        Profile_yTitle = cms.string('Noise from CondDB(ADC)')
    ),
    # -----
    SiStripApvGainsDQM_PSet = cms.PSet(
        Profile_xTitle = cms.string('Apv Number'),
        CondObj_name = cms.string('apvgain'),
        CondObj_fillId = cms.string('ProfileAndCumul'),
        Cumul_xTitle = cms.string('ApvGain from CondDB'),
        Cumul_description = cms.string('ApvGainFromCondDB_Cumul'),
        Cumul_NchX = cms.int32(30),
        Cumul_yTitle = cms.string(' '),
        Profile_description = cms.string('ApvGainFromCondDB_Profile'),
        Cumul_LowX = cms.double(0.0),
        Cumul_HighX = cms.double(1.5),
        Profile_yTitle = cms.string('ApvGain from CondDB')
    ),
    MonitorSiStripPedestal = cms.bool(True),
    # -----
    SiStripQualityDQM_PSet = cms.PSet(
        Profile_xTitle = cms.string('Strip Number'),
        CondObj_name = cms.string('quality'),
        CondObj_fillId = cms.string('onlyProfile'),
        Profile_yTitle = cms.string('Quality Flag from CondDB'),
        Profile_description = cms.string('QualityFlagFromCondDB_Profile')
    ),
    OutputFileName = cms.string('SiStripMonitorCondData.root'),
    MonitorSiStripApvGain = cms.bool(False),
    MonitorSiStripNoise = cms.bool(True),
    MonitorSiStripQuality = cms.bool(False),
    # -----
    SiStripPedestalsDQM_PSet = cms.PSet(
        Profile_xTitle = cms.string('Strip Number'),
        CondObj_name = cms.string('pedestal'),
        CondObj_fillId = cms.string('onlyProfile'),
        SummaryOfProfile_yTitle = cms.string('Pedestal from CondDB(ADC)'),
        SummaryOfProfile_description = cms.string('PedestalFromCondDB_ProfileSummary'),
        Profile_description = cms.string('PedestalFromCondDB_Profile'),
        SummaryOfProfile_xTitle = cms.string('Strip Number'),
        Profile_yTitle = cms.string('Pedestal from CondDB(ADC)')
    ),
    FillConditions_PSet = cms.PSet(
        Mod_On = cms.bool(False),
        # *** exclude a subdetector 
        SubDetectorsToBeExcluded = cms.vstring('none'),
        ModulesToBeFilled = cms.string('all'),
        StripQualityLabel = cms.string('test1'),
        ModulesToBeExcluded = cms.vuint32(),
        # *** exclude OR include a set of modules
        excludeModules = cms.bool(False),
        ModulesToBeIncluded = cms.vuint32(),
        SummaryOnLayerLevel_On = cms.bool(True)
    )
)


