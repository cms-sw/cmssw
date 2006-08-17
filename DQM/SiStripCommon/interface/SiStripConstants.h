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
  static const std::string lldChan_            = "LldChan";
  static const std::string apvPair_            = "ApvPair";
  static const std::string apv_                = "Apv";
  static const std::string unknownGranularity_ = "UnknownGranularity";

  // extra histogram information 
  static const std::string gain_              = "Gain";
  static const std::string digital_           = "Digital";
  static const std::string pedsAndRawNoise_   = "PedsAndRawNoise";
  static const std::string residualsAndNoise_ = "ResidualsAndNoise";
  static const std::string commonMode_        = "CommonMode";

  // summary histogram names (general)
  static const std::string summaryHisto_          = "SummaryHisto";
  static const std::string unknownSummaryHisto_   = "UnknownSummaryHisto";
  static const std::string undefinedSummaryHisto_ = "UndefinedSummaryHisto";
  
  // summary histo names (apv timing)
  static const std::string apvTimingCoarse_ = "ApvTimingCoarse";
  static const std::string apvTimingFine_   = "ApvTimingFine";
  static const std::string apvTimingDelay_  = "apvTimingDelay";
  static const std::string apvTimingError_  = "ApvTimingError";
  static const std::string apvTimingBase_   = "ApvTimingBase";
  static const std::string apvTimingPeak_   = "ApvTimingPeak";
  static const std::string apvTimingHeight_ = "ApvTimingHeight";
  
}

#endif // DQM_SiStripCommon_SiStripConstants_H


