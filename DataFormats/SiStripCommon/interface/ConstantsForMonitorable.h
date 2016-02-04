// Last commit: $Id: ConstantsForMonitorable.h,v 1.14 2010/02/22 14:10:22 wto Exp $

#ifndef DataFormats_SiStripCommon_ConstantsForMonitorable_H
#define DataFormats_SiStripCommon_ConstantsForMonitorable_H

#include "DataFormats/SiStripCommon/interface/Constants.h"
#include <string>

/** 
    @file ConstantsForMonitorable.h

    @brief Constants and enumerated types for sistrip::Monitorable
*/

namespace sistrip { 

  // ---------- Constants ---------- 

  // misc monitorables
  static const char unknownMonitorable_[]   = "UnknownMonitorable";
  static const char undefinedMonitorable_[] = "UndefinedMonitorable";
  
  // fed cabling
  static const char fedCablingFedId_[]    = "FedId";
  static const char fedCablingFedCh_[]    = "FedCh";
  static const char fedCablingAdcLevel_[] = "AdcLevel";

  // fast cabling
  static const char fastCablingDcuId_[]       = "DcuId";
  static const char fastCablingLldCh_[]       = "LldCh";
  static const char fastCablingHighLevel_[]   = "HighLightLevel";
  static const char fastCablingHighRms_[]     = "SpreadInHighLightLevel";
  static const char fastCablingLowLevel_[]    = "LowLightLevel";
  static const char fastCablingLowRms_[]      = "SpreadInLowLightLevel";
  static const char fastCablingAdcRange_[]    = "AdcRangeInLightLevel";
  static const char fastCablingMax_[]         = "MaximumLightLevel";
  static const char fastCablingMin_[]         = "MinimumLightLevel";
  static const char fastCablingConnsPerFed_[] = "ConnectionsPerFed";

  // timing
  static const char apvTimingTime_[]   = "TimeOfTickMarkEdge";
  static const char apvTimingMax_[]    = "MaxSamplingPoint";
  static const char apvTimingDelay_[]  = "RequiredDelayAdjustment";
  static const char apvTimingError_[]  = "ErrorOnTickMarkEdge";
  static const char apvTimingBase_[]   = "TickMarkBase";
  static const char apvTimingPeak_[]   = "TickMarkPeak";
  static const char apvTimingHeight_[] = "TickMarkHeight";

  // timing
  static const char fedTimingTime_[]   = "TimeOfTickMarkEdge";
  static const char fedTimingMax_[]    = "MaxSamplingPoint";
  static const char fedTimingDelay_[]  = "RequiredDelayAdjustment";
  static const char fedTimingError_[]  = "ErrorOnTickMarkEdge";
  static const char fedTimingBase_[]   = "TickMarkBase";
  static const char fedTimingPeak_[]   = "TickMarkPeak";
  static const char fedTimingHeight_[] = "TickMarkHeight";

  // opto scan
  static const char optoScanLldBias_[]     = "LldBiasSetting";
  static const char optoScanLldGain_[]     = "LldGainSetting";
  static const char optoScanMeasGain_[]    = "MeasuredGain";
  static const char optoScanZeroLight_[]   = "ZeroLightLevel";
  static const char optoScanLinkNoise_[]   = "LinkNoise";
  static const char optoScanBaseLiftOff_[] = "BaselineLiftOff";
  static const char optoScanLaserThresh_[] = "LaserThreshold";
  static const char optoScanTickHeight_[]  = "TickHeight";

  // vpsp scan
  static const char vpspScanBothApvs_[]    = "ApvVpspSettings";
  static const char vpspScanApv0_[]        = "Apv0VpspSetting";
  static const char vpspScanApv1_[]        = "Apv1VpspSetting";
  static const char vpspScanAdcLevel_[]    = "BaselineLevel";
  static const char vpspScanDigitalHigh_[] = "DigitalHigh";
  static const char vpspScanDigitalLow_[]  = "DigitalLow";

  // pedestals
  static const char pedestalsAllStrips_[] = "StripPedestals";
  static const char pedestalsMean_[]      = "PedestalMean";
  static const char pedestalsSpread_[]    = "PedestalRmsSpread";
  static const char pedestalsMax_[]       = "PedestalMax";
  static const char pedestalsMin_[]       = "PedestalMin";

  // noise
  static const char noiseAllStrips_[]       = "StripNoise";
  static const char noiseMean_[]            = "NoiseMean";
  static const char noiseSpread_[]          = "NoiseRmsSpread";
  static const char noiseMax_[]             = "NoiseMax";
  static const char noiseMin_[]             = "NoiseMin";
  static const char numOfDead_[]            = "NumOfDeadStrips";
  static const char numOfNoisy_[]           = "NumOfNoisyStrips";
  static const char noiseKSAllStrips_[]     = "StripNoiseKS";
  static const char noiseChi2AllStrips_[]   = "StripNoiseChi2";
  static const char noiseGausAllStrips_[]   = "StripNoiseGaus";
  static const char noiseBin84AllStrips_[]  = "StripNoiseBin84";
  static const char noiseRMSAllStrips_[]  	= "StripNoiseRMS";
  static const char noiseSignif_[]  		= "StripNoiseSignif";
  
  // Fine Delay
  static const char fineDelayPos_[]       = "FineDelayPosition";
  static const char fineDelayErr_[]       = "FineDelayError";

  // Calibration
  static const char calibrationAmplitude_[]    = "CalibrationAmplitude";
  static const char calibrationTail_[]         = "CalibrationTail";
  static const char calibrationRiseTime_[]     = "CalibrationRiseTime";
  static const char calibrationTimeConstant_[] = "CalibrationTimeConstant";
  static const char calibrationSmearing_[]     = "CalibrationSmearing";
  static const char calibrationChi2_[]         = "CalibrationChi2";
  static const char calibrationAmplitudeAS_[]    = "StripCalibrationAmplitude";
  static const char calibrationTailAS_[]         = "StripCalibrationTail";
  static const char calibrationRiseTimeAS_[]     = "StripCalibrationRiseTime";
  static const char calibrationTimeConstantAS_[] = "StripCalibrationTimeConstant";
  static const char calibrationSmearingAS_[]     = "StripCalibrationSmearing";
  static const char calibrationChi2AS_[]         = "StripCalibrationChi2";
  static const char calibrationAmplitudeMin_[]    = "MinCalibrationAmplitude";
  static const char calibrationTailMin_[]         = "MinCalibrationTail";
  static const char calibrationRiseTimeMin_[]     = "MinCalibrationRiseTime";
  static const char calibrationTimeConstantMin_[] = "MinCalibrationTimeConstant";
  static const char calibrationSmearingMin_[]     = "MinCalibrationSmearing";
  static const char calibrationChi2Min_[]         = "MinCalibrationChi2";
  static const char calibrationAmplitudeMax_[]    = "MaxCalibrationAmplitude";
  static const char calibrationTailMax_[]         = "MaxCalibrationTail";
  static const char calibrationRiseTimeMax_[]     = "MaxCalibrationRiseTime";
  static const char calibrationTimeConstantMax_[] = "MaxCalibrationTimeConstant";
  static const char calibrationSmearingMax_[]     = "MaxCalibrationSmearing";
  static const char calibrationChi2Max_[]         = "MaxCalibrationChi2";

  // daq scope mode
  static const char daqScopeModeMeanSignal_[] = "DaqScopeMode_MeanSignal";

  // ---------- Enumerated type ---------- 

  /** Defines the monitorable for the summary histogram. */
  enum Monitorable { UNKNOWN_MONITORABLE   = sistrip::unknown_, 
		     UNDEFINED_MONITORABLE = sistrip::invalid_, 

		     FED_CABLING_FED_ID    = 1301, 
		     FED_CABLING_FED_CH    = 1302, 
		     FED_CABLING_ADC_LEVEL = 1303, 

		     FAST_CABLING_DCU_ID        = 2101, 
		     FAST_CABLING_LLD_CH        = 2102, 
		     FAST_CABLING_HIGH_LEVEL    = 2103, 
		     FAST_CABLING_LOW_LEVEL     = 2104, 
		     FAST_CABLING_HIGH_RMS      = 2105, 
		     FAST_CABLING_LOW_RMS       = 2106, 
		     FAST_CABLING_MAX           = 2107, 
		     FAST_CABLING_MIN           = 2108, 
		     FAST_CABLING_CONNS_PER_FED = 2109, 

		     APV_TIMING_TIME     = 501, 
		     APV_TIMING_MAX_TIME = 502, 
		     APV_TIMING_DELAY    = 503, 
		     APV_TIMING_ERROR    = 504, 
		     APV_TIMING_BASE     = 505, 
		     APV_TIMING_PEAK     = 506, 
		     APV_TIMING_HEIGHT   = 507,

		     FED_TIMING_TIME     = 1201, 
		     FED_TIMING_MAX_TIME = 1202, 
		     FED_TIMING_DELAY    = 1203, 
		     FED_TIMING_ERROR    = 1204, 
		     FED_TIMING_BASE     = 1205, 
		     FED_TIMING_PEAK     = 1206, 
		     FED_TIMING_HEIGHT   = 1207,
		     OPTO_SCAN_LLD_GAIN_SETTING  = 401,
		     OPTO_SCAN_LLD_BIAS_SETTING  = 402,
		     OPTO_SCAN_MEASURED_GAIN     = 403, 
		     OPTO_SCAN_ZERO_LIGHT_LEVEL  = 404, 
		     OPTO_SCAN_LINK_NOISE        = 405,
		     OPTO_SCAN_BASELINE_LIFT_OFF = 406,
		     OPTO_SCAN_LASER_THRESHOLD   = 407,  
		     OPTO_SCAN_TICK_HEIGHT       = 408,

		     VPSP_SCAN_APV_SETTINGS = 1401, 
		     VPSP_SCAN_APV0_SETTING = 1402, 
		     VPSP_SCAN_APV1_SETTING = 1403, 
		     VPSP_SCAN_ADC_LEVEL    = 1404, 
		     VPSP_SCAN_DIGITAL_HIGH = 1405, 
		     VPSP_SCAN_DIGITAL_LOW  = 1406, 

		     PEDESTALS_ALL_STRIPS = 201, 
		     PEDESTALS_MEAN       = 202, 
		     PEDESTALS_SPREAD     = 203, 
		     PEDESTALS_MAX        = 204, 
		     PEDESTALS_MIN        = 205, 

		     NOISE_ALL_STRIPS = 206, 
		     NOISE_MEAN       = 207, 
		     NOISE_SPREAD     = 208, 
		     NOISE_MAX        = 209, 
		     NOISE_MIN        = 210, 
		     NUM_OF_DEAD      = 211, 
		     NUM_OF_NOISY     = 212,
		     NOISE_KS_ALL_STRIPS 	= 215,
		     NOISE_GAUS_ALL_STRIPS  = 216,
		     NOISE_BIN_84_ALL_STRIPS= 217,
		     NOISE_CHI2_ALL_STRIPS 	= 218,
		     NOISE_SIGNIF_ALL_STRIPS= 219,
             NOISE_RMS_ALL_STRIPS 	= 220,
		     FINE_DELAY_POS 		= 601,
		     FINE_DELAY_ERROR 		= 602,

		     CALIBRATION_AMPLITUDE    = 701,
		     CALIBRATION_TAIL         = 702,
		     CALIBRATION_RISETIME     = 703,
		     CALIBRATION_TIMECONSTANT = 704,
		     CALIBRATION_SMEARING     = 705,
		     CALIBRATION_CHI2         = 706,
		     CALIBRATION_AMPLITUDE_ALLSTRIPS    = 707,
		     CALIBRATION_TAIL_ALLSTRIPS         = 708,
		     CALIBRATION_RISETIME_ALLSTRIPS     = 709,
		     CALIBRATION_TIMECONSTANT_ALLSTRIPS = 710,
		     CALIBRATION_SMEARING_ALLSTRIPS     = 711,
		     CALIBRATION_CHI2_ALLSTRIPS         = 712,
		     CALIBRATION_AMPLITUDE_MIN    = 713,
		     CALIBRATION_TAIL_MIN         = 714,
		     CALIBRATION_RISETIME_MIN     = 715,
		     CALIBRATION_TIMECONSTANT_MIN = 716,
		     CALIBRATION_SMEARING_MIN     = 717,
		     CALIBRATION_CHI2_MIN         = 718,
		     CALIBRATION_AMPLITUDE_MAX    = 719,
		     CALIBRATION_TAIL_MAX         = 720,
		     CALIBRATION_RISETIME_MAX     = 721,
		     CALIBRATION_TIMECONSTANT_MAX = 722,
		     CALIBRATION_SMEARING_MAX     = 723,
		     CALIBRATION_CHI2_MAX         = 724,

		     DAQ_SCOPE_MODE_MEAN_SIGNAL = 1501
  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForMonitorable_H
