// Last commit: $Id: ConstantsForMonitorable.h,v 1.4 2007/06/29 10:12:42 bainbrid Exp $

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
  static const std::string unknownMonitorable_   = "UnknownMonitorable";
  static const std::string undefinedMonitorable_ = "UndefinedMonitorable";
  
  // fed cabling
  static const std::string fedCablingFedId_    = "FedId";
  static const std::string fedCablingFedCh_    = "FedCh";
  static const std::string fedCablingAdcLevel_ = "AdcLevel";

  // fast cabling
  static const std::string fastCablingDcuId_       = "DcuId";
  static const std::string fastCablingLldCh_       = "LldCh";
  static const std::string fastCablingHighLevel_   = "HighLightLevel";
  static const std::string fastCablingHighRms_     = "SpreadInHighLightLevel";
  static const std::string fastCablingLowLevel_    = "LowLightLevel";
  static const std::string fastCablingLowRms_      = "SpreadInLowLightLevel";
  static const std::string fastCablingMax_         = "MaximumLightLevel";
  static const std::string fastCablingMin_         = "MinimumLightLevel";
  static const std::string fastCablingConnsPerFed_ = "ConnectionsPerFed";

  // timing
  static const std::string apvTimingTime_   = "TimeOfTickMarkEdge";
  static const std::string apvTimingMax_    = "MaxSamplingPoint";
  static const std::string apvTimingDelay_  = "RequiredDelayAdjustment";
  static const std::string apvTimingError_  = "ErrorOnTickMarkEdge";
  static const std::string apvTimingBase_   = "TickMarkBase";
  static const std::string apvTimingPeak_   = "TickMarkPeak";
  static const std::string apvTimingHeight_ = "TickMarkHeight";

  // timing
  static const std::string fedTimingTime_   = "TimeOfTickMarkEdge";
  static const std::string fedTimingMax_    = "MaxSamplingPoint";
  static const std::string fedTimingDelay_  = "RequiredDelayAdjustment";
  static const std::string fedTimingError_  = "ErrorOnTickMarkEdge";
  static const std::string fedTimingBase_   = "TickMarkBase";
  static const std::string fedTimingPeak_   = "TickMarkPeak";
  static const std::string fedTimingHeight_ = "TickMarkHeight";

  // opto scan
  static const std::string optoScanLldBias_     = "LldBiasSetting";
  static const std::string optoScanLldGain_     = "LldGainSetting";
  static const std::string optoScanMeasGain_    = "MeasuredGain";
  static const std::string optoScanZeroLight_   = "ZeroLightLevel";
  static const std::string optoScanLinkNoise_   = "LinkNoise";
  static const std::string optoScanBaseLiftOff_ = "BaselineLiftOff";
  static const std::string optoScanLaserThresh_ = "LaserThreshold";
  static const std::string optoScanTickHeight_  = "TickMarkHeight";

  // vpsp scan
  static const std::string vpspScanBothApvs_    = "ApvVpspSettings";
  static const std::string vpspScanApv0_        = "Apv0VpspSetting";
  static const std::string vpspScanApv1_        = "Apv1VpspSetting";
  static const std::string vpspScanAdcLevel_    = "BaselineLevel";
  static const std::string vpspScanDigitalHigh_ = "DigitalHigh";
  static const std::string vpspScanDigitalLow_  = "DigitalLow";

  // pedestals
  static const std::string pedestalsAllStrips_ = "StripPedestals";
  static const std::string pedestalsMean_      = "PedestalMean";
  static const std::string pedestalsSpread_    = "PedestalRmsSpread";
  static const std::string pedestalsMax_       = "PedestalMax";
  static const std::string pedestalsMin_       = "PedestalMin";

  // noise
  static const std::string noiseAllStrips_     = "StripNoise";
  static const std::string noiseMean_          = "NoiseMean";
  static const std::string noiseSpread_        = "NoiseRmsSpread";
  static const std::string noiseMax_           = "NoiseMax";
  static const std::string noiseMin_           = "NoiseMin";
  static const std::string numOfDead_          = "NumOfDeadStrips";
  static const std::string numOfNoisy_         = "NumOfNoisyStrips";

  // Fine Delay
  static const std::string fineDelayPos_       = "FineDelayPosition";
  static const std::string fineDelayErr_       = "FineDelayError";

  // daq scope mode
  static const std::string daqScopeModeMeanSignal_ = "DaqScopeMode_MeanSignal";

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

                     FINE_DELAY_POS = 601,
                     FINE_DELAY_ERROR = 602,

		     DAQ_SCOPE_MODE_MEAN_SIGNAL = 1501
  };

}
  
#endif // DataFormats_SiStripCommon_ConstantsForMonitorable_H


