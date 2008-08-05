import FWCore.ParameterSet.Config as cms

CondDataMonitoring = cms.EDFilter("SiStripMonitorCondData",

    OutputFileName             = cms.string('SiStripMonitorCondData.root'),
                                  
    OutputMEsInRootFile        = cms.bool(False),

    MonitorSiStripPedestal     = cms.bool(False),
    MonitorSiStripNoise        = cms.bool(True),
    MonitorSiStripQuality      = cms.bool(False),
    MonitorSiStripApvGain      = cms.bool(False),                                  
    MonitorSiStripLorentzAngle = cms.bool(False),                                  


    FillConditions_PSet = cms.PSet(
    
      Mod_On                  = cms.bool(True),
      SummaryOnStringLevel_On = cms.bool(False),
      SummaryOnLayerLevel_On  = cms.bool(True),

      StripQualityLabel       = cms.string('test1'),
        
      #  exclude OR include a set of modules
      restrictModules         = cms.bool(False),

      ModulesToBeIncluded     = cms.vuint32(), #e.g. {369120277, 369120278, 369120282}
      ModulesToBeExcluded     = cms.vuint32(),
        
      # exclude a subdetector
      SubDetectorsToBeExcluded = cms.vstring('none'), #possibilities : "none" or
                                                      #combinations of {"TIB","TOB","TID","TEC" }
      ModulesToBeFilled = cms.string('all')          
    ),
                                  
    # -----
    SiStripPedestalsDQM_PSet = cms.PSet(

      CondObj_fillId       = cms.string('onlyProfile'),
      CondObj_name         = cms.string('pedestal'),

      FillSummaryAtLayerLevel           = cms.bool(False),
      FillSummaryProfileAtLayerLevel    = cms.bool(False),
      FillCumulativeSummaryAtLayerLevel = cms.bool(True),

      Profile_description     = cms.string('Profile_PedestalFromCondDB'),
      Profile_xTitle          = cms.string('Strip Number'),
      Profile_yTitle          = cms.string('Pedestal from CondDB(ADC)'),
      
      SummaryOfProfile_xTitle = cms.string('Strip Number'),
      SummaryOfProfile_yTitle = cms.string('Pedestal from CondDB(ADC)'),
      SummaryOfProfile_NchY   = cms.int32(100),
      SummaryOfProfile_LowY   = cms.double(0.0),
      SummaryOfProfile_HighY  = cms.double(1000.0),

      Summary_description     = cms.string('Summary_PedestalFromCondDB'),
      Summary_xTitle          = cms.string('detId'),
      Summary_yTitle          = cms.string('Pedestal from CondDB(ADC)'),
      Summary_NchY            = cms.int32(100),
      Summary_LowY            = cms.double(0.0),
      Summary_HighY           = cms.double(1000.0)
    ),

    # -----
    SiStripNoisesDQM_PSet = cms.PSet(

      CondObj_fillId    = cms.string('onlyProfile'),
      CondObj_name      = cms.string('noise'),

      GainRenormalisation               = cms.bool(False),
      FillSummaryAtLayerLevel           = cms.bool(False),
      FillSummaryProfileAtLayerLevel    = cms.bool(False),
      FillCumulativeSummaryAtLayerLevel = cms.bool(True),
      
      Profile_description               = cms.string('Profile_NoiseFromCondDB'),
      Profile_xTitle                    = cms.string('Strip Number'),
      Profile_yTitle                    = cms.string('Noise from CondDB(ADC)'),

      Cumul_description = cms.string('NoiseFromCondDB'),
      Cumul_xTitle      = cms.string('Noise from CondDB(ADC)'),
      Cumul_yTitle      = cms.string(' '),
      Cumul_NchX        = cms.int32(50),
      Cumul_LowX        = cms.double(0.0),
      Cumul_HighX       = cms.double(5.0),
      
      SummaryOfProfile_description = cms.string('ProfileSummary_NoiseFromCondDB'),
      SummaryOfProfile_xTitle      = cms.string('Strip Number'),      
      SummaryOfProfile_yTitle      = cms.string('Noise from CondDB(ADC)'),
      SummaryOfProfile_NchY        = cms.int32(50),
      SummaryOfProfile_LowY        = cms.double(0.0),
      SummaryOfProfile_HighY       = cms.double(5.0),

      Summary_description          = cms.string('Summary_NoiseFromCondDB'),
      Summary_xTitle               = cms.string('detId'),
      Summary_yTitle               = cms.string('Noise from CondDB(ADC)'),
      Summary_NchY                 = cms.int32(50),
      Summary_LowY                 = cms.double(0.0),
      Summary_HighY                = cms.double(5.0),
      
      SummaryOfCumul_description = cms.string('CumulativeSummary_NoiseFromCondDB'),
      SummaryOfCumul_xTitle      = cms.string('Noise from CondDB'),
      SummaryOfCumul_yTitle      = cms.string(' '),
      SummaryOfCumul_NchX        = cms.int32(50),
      SummaryOfCumul_LowX        = cms.double(0.0),
      SummaryOfCumul_HighX       = cms.double(10.0)
    ),

    # -----
    SiStripQualityDQM_PSet = cms.PSet(

      CondObj_name   = cms.string('quality'),
      CondObj_fillId = cms.string('onlyProfile'),
      
      FillSummaryAtLayerLevel           = cms.bool(False),
      FillSummaryProfileAtLayerLevel    = cms.bool(False),
      FillCumulativeSummaryAtLayerLevel = cms.bool(True),

      Profile_description = cms.string('Profile_QualityFlagFromCondDB'),
      Profile_xTitle      = cms.string('Strip Number'),
      Profile_yTitle      = cms.string('Quality Flag from CondDB'),
      
      Summary_description = cms.string('Summary_FractionOfBadStripsFromCondDB'),
      Summary_xTitle      = cms.string('detId'),
      Summary_yTitle      = cms.string('Fraction of bad strips from CondDB(%)'),
      Summary_NchY        = cms.int32(100),
      Summary_LowY        = cms.double(0.5),
      Summary_HighY       = cms.double(100.5)
    ),

    # -----
    SiStripApvGainsDQM_PSet = cms.PSet(
    
      CondObj_name   = cms.string('apvgain'),
      CondObj_fillId = cms.string('ProfileAndCumul'),

      FillSummaryAtLayerLevel           = cms.bool(False),
      FillSummaryProfileAtLayerLevel    = cms.bool(False),
      FillCumulativeSummaryAtLayerLevel = cms.bool(True),

      Profile_description = cms.string('Profile_ApvGainFromCondDB'),
      Profile_xTitle      = cms.string('Apv Number'),
      Profile_yTitle      = cms.string('ApvGain from CondDB'),

      Cumul_description   = cms.string('ApvGainFromCondDB'),
      Cumul_xTitle        = cms.string('ApvGain from CondDB'),
      Cumul_yTitle        = cms.string(' '),        
      Cumul_NchX          = cms.int32(50),
      Cumul_LowX          = cms.double(0.5),
      Cumul_HighX         = cms.double(1.5),

      SummaryOfProfile_description = cms.string('ProfileSummary_ApvGainFromCondDB'),
      SummaryOfProfile_xTitle      = cms.string('Apv Number'),
      SummaryOfProfile_yTitle      = cms.string('ApvGain from CondDB'),
      SummaryOfProfile_NchY        = cms.int32(50),
      SummaryOfProfile_LowY        = cms.double(0.5),
      SummaryOfProfile_HighY       = cms.double(1.5),

      Summary_description   = cms.string('Summary_ApvGainFromCondDB'),
      Summary_xTitle        = cms.string('detId'),
      Summary_yTitle        = cms.string('ApvGain from CondDB'),
      Summary_NchY          = cms.int32(50),        
      Summary_LowY          = cms.double(0.5),
      Summary_HighY         = cms.double(1.5)
    ),

    # -----
    SiStripLorentzAngleDQM_PSet = cms.PSet(

      CondObj_name = cms.string('lorentzangle'),
      CondObj_fillId = cms.string('none'),
      
      FillSummaryAtLayerLevel = cms.bool(False),
      FillSummaryProfileAtLayerLevel = cms.bool(False),
      FillCumulativeSummaryAtLayerLevel = cms.bool(True),

      SummaryOfCumul_description = cms.string('ProfileSummary_LorentzAngleFromCondDB'),
      SummaryOfCumul_xTitle      = cms.string('LorentzAngle from CondDB'),
      SummaryOfCumul_yTitle      = cms.string(' '),
      SummaryOfCumul_NchX        = cms.int32(50),      
      SummaryOfCumul_LowX        = cms.double(0.01),
      SummaryOfCumul_HighX       = cms.double(0.06),
      
      SummaryOfProfile_description = cms.string('Summary_LorentzAngleFromCondDB'),
      SummaryOfProfile_xTitle      = cms.string('detId'),
      SummaryOfProfile_yTitle      = cms.string('LorentzAngle from CondDB'),
      SummaryOfProfile_NchY        = cms.int32(50),
      SummaryOfProfile_LowY        = cms.double(0.01),
      SummaryOfProfile_HighY       = cms.double(0.06)      
    )
)
