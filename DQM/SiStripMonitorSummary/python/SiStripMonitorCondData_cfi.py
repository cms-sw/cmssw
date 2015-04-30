import FWCore.ParameterSet.Config as cms

CondDataMonitoring = cms.EDAnalyzer("SiStripMonitorCondData",

    OutputFileName             = cms.string('SiStripMonitorCondData_MYTEST.root'),
                                  
    OutputMEsInRootFile        = cms.bool(True),

    MonitorSiStripPedestal     = cms.bool(True),
    MonitorSiStripNoise        = cms.bool(True),
    MonitorSiStripQuality      = cms.bool(False),
    MonitorSiStripCabling      = cms.bool(True),
    MonitorSiStripLowThreshold = cms.bool(True),
    MonitorSiStripHighThreshold= cms.bool(True),
    MonitorSiStripApvGain      = cms.bool(True),                              
    MonitorSiStripLorentzAngle = cms.bool(True),                            
    MonitorSiStripBackPlaneCorrection = cms.bool(True),                            

    FillConditions_PSet = cms.PSet(
      FolderName_For_QualityAndCabling_SummaryHistos= cms.string("SiStrip/Tracks"),
      OutputSummaryAtLayerLevelAsImage           = cms.bool(True),
      OutputSummaryProfileAtLayerLevelAsImage    = cms.bool(True),
      OutputCumulativeSummaryAtLayerLevelAsImage = cms.bool(True),
      Mod_On                  = cms.bool(False),
      HistoMaps_On            = cms.bool(True),
      SummaryOnStringLevel_On = cms.bool(False),
      SummaryOnLayerLevel_On  = cms.bool(True),
      GrandSummary_On         = cms.bool(True),
      StripQualityLabel       = cms.string(''),

      ActiveDetIds_On         =  cms.bool(False),
      TkMap_On                =  cms.bool(False),

      #  exclude OR include a set of modules
      restrictModules         = cms.bool(False),
      ModulesToBeIncluded_DetIdSelector = cms.vstring(
#            "0x1e000000-0x16000000",  #TIB
#            "0x1e000000-0x18000000",  #TID
#            "0x1e006000-0x18002000", #TIDm
#            "0x1e006000-0x18004000", #TIDp
#            "0x1e000000-0x1a000000", #TOB
#            "0x1e000000-0x1c000000", #TEC
#            "0x1e0c0000-0x1c040000", #TECm
#            "0x1e0c0000-0x1c080000"  #TECp
      ),
      ModulesToBeExcluded_DetIdSelector = cms.vstring(
#            "0x1e000000-0x16000000",  #TIB
#            "0x1e000000-0x18000000",  #TID
#            "0x1e006000-0x18002000", #TIDm
#            "0x1e006000-0x18004000", #TIDp
#            "0x1e000000-0x1a000000", #TOB
#            "0x1e000000-0x1c000000", #TEC
#            "0x1e0c0000-0x1c040000", #TECm
#            "0x1e0c0000-0x1c080000"  #TECp
      ),
      ModulesToBeIncluded     = cms.vuint32(), #e.g. {369120277, 369120278, 369120282}
      ModulesToBeExcluded     = cms.vuint32(),
        
      # exclude a subdetector
      SubDetectorsToBeExcluded = cms.vstring('none'), #possibilities : "none" or
                                                      #combinations of {"TIB","TOB","TID","TEC" }
      ModulesToBeFilled = cms.string('all')          
    ),
                                  

    # -----

    SiStripCablingDQM_PSet = cms.PSet(

    ActiveDetIds_On         =  cms.bool(True),

    TkMap_On                =  cms.bool(True),
    TkMapName               =  cms.string('CablingTkMap.png'),
    minValue               =  cms.double(0.),
    maxValue               =  cms.double(6.), 
    saturatedFraction      = cms.double(.01),
    
    CondObj_fillId       = cms.string('ProfileAndCumul'),
    CondObj_name         = cms.string('fedcabling')
    ),    

    # -----

    SiStripPedestalsDQM_PSet = cms.PSet(

      ActiveDetIds_On         =  cms.bool(True),
      
      TkMap_On                =  cms.bool(True),     
      TkMapName               =  cms.string('PedestalTkMap.png'),
      minValue               =  cms.double(0.),
      maxValue               =  cms.double(400.),  
      saturatedFraction      = cms.double(.01),

      CondObj_fillId       = cms.string('onlyProfile'),
      CondObj_name         = cms.string('pedestal'),

      FillSummaryAtLayerLevel           = cms.bool(True),
      FillSummaryProfileAtLayerLevel    = cms.bool(True),

      Profile_description     = cms.string('Profile_PedestalFromCondDB'),
      Profile_xTitle          = cms.string('Strip Number'),
      Profile_yTitle          = cms.string('Pedestal from CondDB(ADC)'),
      
      SummaryOfProfile_description = cms.string('ProfileSummary_PedestalFromCondDB'),
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

      ActiveDetIds_On         =  cms.bool(True),
      
      TkMap_On                =  cms.bool(True),
      TkMapName               =  cms.string('NoiseTkMap.png'),
      minValue               =  cms.double(3.),
      maxValue               =  cms.double(9.),      
      saturatedFraction      = cms.double(.01),

      CondObj_fillId    = cms.string('onlyCumul'),
      CondObj_name      = cms.string('noise'),

      GainRenormalisation               = cms.bool(False),
      SimGainRenormalisation               = cms.bool(False),
      
      FillSummaryAtLayerLevel           = cms.bool(True),
      FillSummaryProfileAtLayerLevel    = cms.bool(True),
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
      SummaryOfProfile_HighY       = cms.double(10.0),

      Summary_description          = cms.string('Summary_NoiseFromCondDB'),
      Summary_xTitle               = cms.string('detId'),
      Summary_yTitle               = cms.string('Noise from CondDB(ADC)'),
      Summary_NchY                 = cms.int32(50),
      Summary_LowY                 = cms.double(0.0),
      Summary_HighY                = cms.double(10.0),
      
      SummaryOfCumul_description = cms.string('CumulativeSummary_NoiseFromCondDB'),
      SummaryOfCumul_xTitle      = cms.string('Noise from CondDB'),
      SummaryOfCumul_yTitle      = cms.string(' '),
      SummaryOfCumul_NchX        = cms.int32(50),
      SummaryOfCumul_LowX        = cms.double(0.0),
      SummaryOfCumul_HighX       = cms.double(10.0)
    ),

    # -----
    SiStripQualityDQM_PSet = cms.PSet(

      ActiveDetIds_On         =  cms.bool(True),
    
      TkMap_On                =  cms.bool(True),
      TkMapName               =  cms.string('QualityTkMap.svg'),
      minValue               =  cms.double(0.),
      maxValue               =  cms.double(0.), 
      saturatedFraction      = cms.double(.01),


      CondObj_name   = cms.string('quality'),
      CondObj_fillId = cms.string('onlyProfile'),
      
      FillSummaryAtLayerLevel           = cms.bool(True),

      Profile_description = cms.string('Profile_QualityFlagFromCondDB'),
      Profile_xTitle      = cms.string('Strip Number'),
      Profile_yTitle      = cms.string('Quality Flag from CondDB'),
      
      Summary_description = cms.string('Summary_FractionOfBadStripsFromCondDB'),
      Summary_xTitle      = cms.string('detId'),
      Summary_yTitle      = cms.string('Fraction of bad strips from CondDB(%)'),
      Summary_NchY        = cms.int32(100),
      Summary_LowY        = cms.double(0.5),
      Summary_HighY       = cms.double(100.5),

      Summary_BadObjects_histo_xTitle      =cms.string('Sub Det And Layer'),
      
      Summary_BadModules_histo_name =cms.string('Summary_BadModules_FromCondDB'),
      Summary_BadModules_histo_yTitle      =cms.string('Number of bad Modules from CondDB'),
      
      Summary_BadFibers_histo_name =cms.string('Summary_BadFibers_FromCondDB'),
      Summary_BadFibers_histo_yTitle      =cms.string('Number of bad Fibers from CondDB'),
      
      Summary_BadApvs_histo_name =cms.string('Summary_BadApvs_FromCondDB'),
      Summary_BadApvs_histo_yTitle      =cms.string('Number of bad Apvs from CondDB'),
      
      Summary_BadStrips_histo_name =cms.string('Summary_BadStrips_FromCondDB'),
      Summary_BadStrips_histo_yTitle      =cms.string('Number of bad Strips from CondDB'),
      
      SummaryOfCumul_description   =cms.string('CumulativeSummary_SiStripQualityFromCondDB'),
      SummaryOfCumul_xTitle        =cms.string('SiStripQualityfrom CondDB'),
      SummaryOfCumul_yTitle        =cms.string(' '),
      
      SummaryOfCumul_NchX          = cms.int32(100),
      SummaryOfCumul_LowX          = cms.double(0.0),
      SummaryOfCumul_HighX         = cms.double(100.0)	

    ),

    # -----
    SiStripApvGainsDQM_PSet = cms.PSet(

      ActiveDetIds_On         =  cms.bool(True),

      TkMap_On                =  cms.bool(True),
      TkMapName               =  cms.string('GainTkMap.svg'),
      minValue               =  cms.double(0.),
      maxValue               =  cms.double(1.5),  
      saturatedFraction      = cms.double(.01),

      CondObj_name   = cms.string('apvgain'),
      CondObj_fillId = cms.string('ProfileAndCumul'),

      FillSummaryAtLayerLevel           = cms.bool(True),
      FillSummaryProfileAtLayerLevel    = cms.bool(True),

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

    ActiveDetIds_On         =  cms.bool(False),
      
    TkMap_On                =  cms.bool(True),
    TkMapName               =  cms.string('LorentzAngleTkMap.png'),
    minValue               =  cms.double(0.01),
    maxValue               =  cms.double(0.03),
    saturatedFraction      = cms.double(.01),

      CondObj_name = cms.string('lorentzangle'),
      CondObj_fillId = cms.string('ProfileAndCumul'),
      
      FillSummaryProfileAtLayerLevel = cms.bool(True),
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
    ),

    # -----
    SiStripBackPlaneCorrectionDQM_PSet = cms.PSet(

    ActiveDetIds_On         =  cms.bool(False),
      
    TkMap_On                =  cms.bool(True),
    TkMapName               =  cms.string('BackPlaneCorrectionTkMap.png'),
    minValue               =  cms.double(0.00),
    maxValue               =  cms.double(0.10),
    saturatedFraction      = cms.double(.01),

      CondObj_name = cms.string('bpcorrection'),
      CondObj_fillId = cms.string('ProfileAndCumul'),
      
      FillSummaryProfileAtLayerLevel = cms.bool(True),
      FillCumulativeSummaryAtLayerLevel = cms.bool(True),

      SummaryOfCumul_description = cms.string('ProfileSummary_BackPlaneCorrectionFromCondDB'),
      SummaryOfCumul_xTitle      = cms.string('BackPlaneCorrection from CondDB'),
      SummaryOfCumul_yTitle      = cms.string(' '),
      SummaryOfCumul_NchX        = cms.int32(50),      
      SummaryOfCumul_LowX        = cms.double(0.00),
      SummaryOfCumul_HighX       = cms.double(0.10),
      
      SummaryOfProfile_description = cms.string('Summary_BackPlaneCorrectionFromCondDB'),
      SummaryOfProfile_xTitle      = cms.string('detId'),
      SummaryOfProfile_yTitle      = cms.string('BackPlaneCorrection from CondDB'),
      SummaryOfProfile_NchY        = cms.int32(50),
      SummaryOfProfile_LowY        = cms.double(0.00),
      SummaryOfProfile_HighY       = cms.double(0.10)      
    ),

    # -----

      SiStripLowThresholdDQM_PSet = cms.PSet(

      ActiveDetIds_On         =  cms.bool(True),
      
      TkMap_On                =  cms.bool(True),
      TkMapName               =  cms.string('LowThresholdTkMap.png'),
      minValue               =  cms.double(0.),
      maxValue               =  cms.double(10.), 
      saturatedFraction      = cms.double(.01),

      WhichThreshold= cms.string('Low'),

      CondObj_fillId = cms.string('onlyProfile'), 
      CondObj_name   = cms.string('lowthreshold'),  

      FillSummaryAtLayerLevel= cms.bool(True),
      FillSummaryProfileAtLayerLevel=cms.bool(True),
      FillCumulativeSummaryAtLayerLevel = cms.bool(True),

      Profile_description = cms.string('Profile_LowThresholdFromCondDB'),
      Profile_xTitle      = cms.string('Strip Number'),
      Profile_yTitle      = cms.string('Low Threshold from CondDB(ADC)'),
      
      SummaryOfProfile_description = cms.string('ProfileSummary_LowThresholdFromCondDB'),
      SummaryOfProfile_xTitle      = cms.string('Strip Number'),
      SummaryOfProfile_yTitle      = cms.string('Low Threshold from CondDB(ADC)'),	
      SummaryOfProfile_NchY        = cms.int32(100),
      SummaryOfProfile_LowY        = cms.double(0),
      SummaryOfProfile_HighY       = cms.double(10),

      Summary_description   = cms.string('Summary_LowThresholdFromCondDB'),
      Summary_xTitle        = cms.string('detId'),
      Summary_yTitle        = cms.string('Low Threshold from CondDB(ADC)'),
      Summary_NchY          = cms.int32(100),
      Summary_LowY          = cms.double(0),
      Summary_HighY         = cms.double(10)
      ),


    # -----

    SiStripHighThresholdDQM_PSet = cms.PSet(

    ActiveDetIds_On         =  cms.bool(True),

    TkMap_On                =  cms.bool(True),
    TkMapName               =  cms.string('HighThresholdTkMap.png'),
    minValue               =  cms.double(0.),
    maxValue               =  cms.double(10.), 	
    saturatedFraction      = cms.double(.01),

    WhichThreshold= cms.string('High'),
    
    CondObj_fillId = cms.string('onlyProfile'), 
    CondObj_name   = cms.string('highthreshold'),  

    FillSummaryAtLayerLevel= cms.bool(True),
    FillSummaryProfileAtLayerLevel=cms.bool(True),
    FillCumulativeSummaryAtLayerLevel = cms.bool(True),

    Profile_description = cms.string('Profile_HighThresholdFromCondDB'),
    Profile_xTitle      = cms.string('Strip Number'),
    Profile_yTitle      = cms.string('High Threshold from CondDB(ADC)'),
	
    SummaryOfProfile_description = cms.string('ProfileSummary_HighThresholdFromCondDB'),
    SummaryOfProfile_xTitle      = cms.string('Strip Number'),
    SummaryOfProfile_yTitle      = cms.string('High Threshold from CondDB(ADC)'),	
    SummaryOfProfile_NchY        = cms.int32(100),
    SummaryOfProfile_LowY        = cms.double(0),
    SummaryOfProfile_HighY       = cms.double(10),
	
    Summary_description   = cms.string('Summary_HighThresholdFromCondDB'),
    Summary_xTitle        = cms.string('detId'),
    Summary_yTitle        = cms.string('High Threshold from CondDB(ADC)'),
    Summary_NchY          = cms.int32(100),
    Summary_LowY          = cms.double(0),
    Summary_HighY         = cms.double(10)
    )
                                  
)
