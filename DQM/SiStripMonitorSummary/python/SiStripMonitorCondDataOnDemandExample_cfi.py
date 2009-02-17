# The following comments couldn't be translated into the new config version:

# combinations of {"TIB","TOB","TID","TEC" }
# *** other restricting specifications

import FWCore.ParameterSet.Config as cms

myOnDemandExample = cms.EDFilter("SiStripMonitorCondDataOnDemandExample",
    OutputMEsInRootFile = cms.bool(False),
    MonitorSiStripLorentzAngle = cms.bool(True),
    # -----
    SiStripApvGainsDQM_PSet = cms.PSet(
        Profile_xTitle = cms.string('Apv Number'),
        Summary_LowY = cms.double(0.5),
        Summary_xTitle = cms.string('detId'),
        Cumul_HighX = cms.double(1.5),
        Cumul_description = cms.string('ApvGainFromCondDB'),
        SummaryOfProfile_yTitle = cms.string('ApvGain from CondDB'),
        SummaryOfProfile_NchY = cms.int32(50),
        Summary_description = cms.string('Summary_ApvGainFromCondDB'),
        SummaryOfProfile_LowY = cms.double(0.5),
        Summary_yTitle = cms.string('ApvGain from CondDB'),
        SummaryOfProfile_HighY = cms.double(1.5),
        Cumul_NchX = cms.int32(50),
        CondObj_name = cms.string('apvgain'),
        CondObj_fillId = cms.string('ProfileAndCumul'),
        Cumul_xTitle = cms.string('ApvGain from CondDB'),
        Summary_HighY = cms.double(1.5),
        SummaryOfProfile_description = cms.string('ProfileSummary_ApvGainFromCondDB'),
        Summary_NchY = cms.int32(50),
        Cumul_yTitle = cms.string(' '),
        Profile_description = cms.string('Profile_ApvGainFromCondDB'),
        SummaryOfProfile_xTitle = cms.string('Apv Number'),
        Cumul_LowX = cms.double(0.5),
        Profile_yTitle = cms.string('ApvGain from CondDB')
    ),
    MonitorSiStripPedestal = cms.bool(True),
    # -----
    SiStripQualityDQM_PSet = cms.PSet(
        Profile_xTitle = cms.string('Strip Number'),
        CondObj_name = cms.string('quality'),
        CondObj_fillId = cms.string('onlyProfile'),
        Summary_HighY = cms.double(100.5),
        Summary_xTitle = cms.string('detId'),
        Summary_NchY = cms.int32(100),
        Summary_description = cms.string('Summary_FractionOfBadStripsFromCondDB'),
        Profile_description = cms.string('Profile_QualityFlagFromCondDB'),
        Summary_yTitle = cms.string('Fraction of bad strips from CondDB(%)'),
        Summary_LowY = cms.double(0.5),
        Profile_yTitle = cms.string('Quality Flag from CondDB')
    ),
    OutputFileName = cms.string('SiStripMonitorCondDataOnDemandExample.root'),
    MonitorSiStripApvGain = cms.bool(True),
    # -----
    SiStripLorentzAngleDQM_PSet = cms.PSet(
        SummaryOfCumul_NchX = cms.int32(50),
        SummaryOfCumul_xTitle = cms.string('LorentzAngle from CondDB'),
        CondObj_name = cms.string('lorentzangle'),
        CondObj_fillId = cms.string('none'),
        SummaryOfCumul_LowX = cms.double(0.01),
        SummaryOfCumul_HighX = cms.double(0.06),
        SummaryOfProfile_yTitle = cms.string('LorentzAngle from CondDB'),
        SummaryOfProfile_description = cms.string('Summary_LorentzAngleFromCondDB'),
        SummaryOfCumul_yTitle = cms.string(' '),
        SummaryOfCumul_description = cms.string('ProfileSummary_LorentzAngleFromCondDB'),
        SummaryOfProfile_LowY = cms.double(0.01),
        SummaryOfProfile_xTitle = cms.string('detId'),
        SummaryOfProfile_NchY = cms.int32(50),
        SummaryOfProfile_HighY = cms.double(0.06)
    ),
    MonitorSiStripNoise = cms.bool(True),
    MonitorSiStripQuality = cms.bool(True),
    # -----
    SiStripNoisesDQM_PSet = cms.PSet(
        Profile_xTitle = cms.string('Strip Number'),
        Summary_LowY = cms.double(0.0),
        Summary_xTitle = cms.string('detId'),
        Cumul_HighX = cms.double(5.0),
        Cumul_description = cms.string('NoiseFromCondDB'),
        SummaryOfProfile_yTitle = cms.string('Noise from CondDB(ADC)'),
        SummaryOfProfile_NchY = cms.int32(50),
        GainRenormalisation = cms.bool(True),
        Summary_description = cms.string('Summary_NoiseFromCondDB'),
        SummaryOfProfile_LowY = cms.double(0.0),
        Summary_yTitle = cms.string('Noise from CondDB(ADC)'),
        SummaryOfProfile_HighY = cms.double(5.0),
        Cumul_NchX = cms.int32(50),
        CondObj_name = cms.string('noise'),
        CondObj_fillId = cms.string('ProfileAndCumul'),
        Cumul_xTitle = cms.string('Noise from CondDB(ADC)'),
        Summary_HighY = cms.double(5.0),
        SummaryOfProfile_description = cms.string('ProfileSummary_NoiseFromCondDB'),
        Summary_NchY = cms.int32(50),
        Cumul_yTitle = cms.string(' '),
        Profile_description = cms.string('Profile_NoiseFromCondDB'),
        SummaryOfProfile_xTitle = cms.string('Strip Number'),
        Cumul_LowX = cms.double(0.0),
        Profile_yTitle = cms.string('Noise from CondDB(ADC)')
    ),
    # -----
    SiStripPedestalsDQM_PSet = cms.PSet(
        Profile_xTitle = cms.string('Strip Number'),
        CondObj_name = cms.string('pedestal'),
        CondObj_fillId = cms.string('onlyProfile'),
        Summary_HighY = cms.double(1000.0),
        SummaryOfProfile_yTitle = cms.string('Pedestal from CondDB(ADC)'),
        SummaryOfProfile_description = cms.string('ProfileSummary_PedestalFromCondDB'),
        Summary_NchY = cms.int32(100),
        Summary_description = cms.string('Summary_PedestalFromCondDB'),
        SummaryOfProfile_LowY = cms.double(0.0),
        SummaryOfProfile_xTitle = cms.string('Strip Number'),
        Profile_description = cms.string('Profile_PedestalFromCondDB'),
        Summary_yTitle = cms.string('Pedestal from CondDB(ADC)'),
        SummaryOfProfile_NchY = cms.int32(100),
        SummaryOfProfile_HighY = cms.double(1000.0),
        Profile_yTitle = cms.string('Pedestal from CondDB(ADC)'),
        Summary_LowY = cms.double(0.0),
        Summary_xTitle = cms.string('detId')
    ),
    FillConditions_PSet = cms.PSet(
        Mod_On = cms.bool(False),
        # *** exclude a subdetector 
        SubDetectorsToBeExcluded = cms.vstring('none'),
        ModulesToBeFilled = cms.string('all'),
        StripQualityLabel = cms.string('test1'),
        ModulesToBeIncluded = cms.vuint32(), ##e.g. {369120277, 369120278, 369120282}

        ModulesToBeExcluded = cms.vuint32(),
        # *** exclude OR include a set of modules
        restrictModules = cms.bool(False),
        SummaryOnStringLevel_On = cms.bool(False),
        SummaryOnLayerLevel_On = cms.bool(True)
    )
)


