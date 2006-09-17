#ifndef DQM_SiStripCommon_SiStripConstants_H
#define DQM_SiStripCommon_SiStripConstants_H

#include "boost/cstdint.hpp"
#include <string>

// -----------------------------------------------------------------------------
// Constants associated with the naming of DQM histograms
namespace sistrip { 
  
  // generic constants
  static const std::string root_ = "SiStrip";
  static const std::string dir_  = "/";
  static const std::string sep_  = "_";
  static const std::string pipe_ = "|";
  static const std::string dot_  = ".";
  static const std::string commissioningTask_ = "SiStripCommissioningTask";
  
  // views
  static const std::string controlView_  = "ControlView";
  static const std::string readoutView_  = "ReadoutView";
  static const std::string detectorView_ = "DetectorView";
  static const std::string unknownView_  = "UnknownView";

  // control and readout parameters
  static const std::string fecCrate_    = "FecCrate";
  static const std::string fecSlot_     = "FecSlot";
  static const std::string fecRing_     = "FecRing";
  static const std::string ccuAddr_     = "CcuAddr";
  static const std::string ccuChan_     = "CcuChan";
  static const std::string fedId_       = "FedId";
  static const std::string feUnit_      = "FedFeUnit";
  static const std::string fedFeChan_   = "FedFeChan";
  static const std::string fedChannel_  = "FedChannel";

  // commissioning task
  static const std::string fedCabling_    = "FedCabling";
  static const std::string apvTiming_     = "ApvTiming";
  static const std::string fedTiming_     = "FedTiming";
  static const std::string optoScan_      = "OptoScan";
  static const std::string vpspScan_      = "VpspScan";
  static const std::string pedestals_     = "Pedestals";
  static const std::string apvLatency_    = "ApvLatency";
  static const std::string undefinedTask_ = "UndefinedTask";
  static const std::string unknownTask_   = "UnknownTask";

  // histo contents
  static const std::string sum2_            = "SumOfSquares";
  static const std::string sum_             = "SumOfContents";
  static const std::string num_             = "NumOfEntries";
  static const std::string unknownContents_ = "UnknownContents";

  // key 
  static const std::string fedKey_     = "FedKey";
  static const std::string fecKey_     = "FecKey";
  static const std::string detKey_     = "DetId"; //@@ necessary?
  static const std::string unknownKey_ = "UnknownKey";

  // granularity
  static const std::string tracker_            = "Tracker";
  static const std::string partition_          = "Partition";
  static const std::string module_             = "Module";
  static const std::string lldChan_            = "LldChan";
  //static const std::string apvPair_            = "ApvPair";
  static const std::string apv_                = "Apv";
  static const std::string unknownGranularity_ = "UnknownGranularity";

  // extra histogram information 
  static const std::string gain_              = "Gain";
  static const std::string digital_           = "Digital";
  static const std::string pedsAndRawNoise_   = "PedsAndRawNoise";
  static const std::string residualsAndNoise_ = "ResidualsAndNoise";
  static const std::string commonMode_        = "CommonMode";

  // summary histogram types
  static const std::string summaryDistr_         = "SummaryDistr";
  static const std::string summary1D_            = "Summary1D";
  static const std::string summary2D_            = "Summary2D";
  static const std::string summaryProf_          = "SummaryProfile";
  static const std::string unknownSummaryType_   = "UnknownSummaryType";
  static const std::string undefinedSummaryType_ = "UndefinedSummaryType";

  // summary histogram names (general)
  static const std::string summaryHisto_          = "SummaryHisto";
  static const std::string unknownSummaryHisto_   = "UnknownSummaryHisto";
  static const std::string undefinedSummaryHisto_ = "UndefinedSummaryHisto";
  
  // summary histo names (apv timing)
  static const std::string apvTimingTime_   = "TimingDelay";
  static const std::string apvTimingMax_    = "ApvTimingMax";
  static const std::string apvTimingDelay_  = "ApvTimingDelay";
  static const std::string apvTimingError_  = "ApvTimingError";
  static const std::string apvTimingBase_   = "ApvTimingBase";
  static const std::string apvTimingPeak_   = "ApvTimingPeak";
  static const std::string apvTimingHeight_ = "ApvTimingHeight";

  // summary histo names (fed timing)
  static const std::string fedTimingTime_   = "FedTimingTime";
  static const std::string fedTimingMax_    = "FedTimingMax";
  static const std::string fedTimingDelay_  = "FedTimingDelay";
  static const std::string fedTimingError_  = "FedTimingError";
  static const std::string fedTimingBase_   = "FedTimingBase";
  static const std::string fedTimingPeak_   = "FedTimingPeak";
  static const std::string fedTimingHeight_ = "FedTimingHeight";

  // summary histo names (opto scan)
  static const std::string optoScanLldBias_     = "OptoScanLldBias";
  static const std::string optoScanLldGain_     = "OptoScanLldGain";
  static const std::string optoScanMeasGain_    = "OptoScanMeasuredGain";
  static const std::string optoScanZeroLight_   = "OptoScanZeroLight";
  static const std::string optoScanLinkNoise_   = "OptoScanLinkNoise";
  static const std::string optoScanBaseLiftOff_ = "OptoScanBaseLiftOff";
  static const std::string optoScanLaserThresh_ = "OptoScanLaserThresh";
  static const std::string optoScanTickHeight_  = "OptoScanTickHeight";

  // summary histo names (vpsp scan)
  static const std::string vpspScanBothApvs_ = "VpspScanBothApvs";
  static const std::string vpspScanApv0_ = "VpspScanApv0";
  static const std::string vpspScanApv1_ = "VpspScanApv1";

  // summary histo names (pedestals)
  static const std::string sistrip::pedestalsAllStrips_ = "Pedestals_AllStrips";
  static const std::string sistrip::pedestalsMean_      = "Pedestals_Mean";
  static const std::string sistrip::pedestalsSpread_    = "Pedestals_Spread";
  static const std::string sistrip::pedestalsMax_       = "Pedestals_Max";
  static const std::string sistrip::pedestalsMin_       = "Pedestals_Min";
  static const std::string sistrip::noiseAllStrips_     = "Noise_AllStrips";
  static const std::string sistrip::noiseMean_          = "Noise_Mean";
  static const std::string sistrip::noiseSpread_        = "Noise_Spread";
  static const std::string sistrip::noiseMax_           = "Noise_Max";
  static const std::string sistrip::noiseMin_           = "Noise_Min";
  static const std::string sistrip::numOfDead_          = "NumOfDead_Strips";
  static const std::string sistrip::numOfNoisy_         = "NumOfNoisy_Strips";
  
}

#endif // DQM_SiStripCommon_SiStripConstants_H


