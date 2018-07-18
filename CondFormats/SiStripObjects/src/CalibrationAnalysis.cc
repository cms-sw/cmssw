#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
CalibrationAnalysis::CalibrationAnalysis( const uint32_t& key, const bool& deconv, int calchan ) 
  : CommissioningAnalysis(key,"CalibrationAnalysis"),
    amplitude_(2,VFloat(128,sistrip::invalid_)),
    tail_(2,VFloat(128,sistrip::invalid_)),
    riseTime_(2,VFloat(128,sistrip::invalid_)),
    timeConstant_(2,VFloat(128,sistrip::invalid_)),
    turnOn_(2,VFloat(128,sistrip::invalid_)),
    maximum_(2,VFloat(128,sistrip::invalid_)),
    undershoot_(2,VFloat(128,sistrip::invalid_)),
    baseline_(2,VFloat(128,sistrip::invalid_)),
    smearing_(2,VFloat(128,sistrip::invalid_)),
    chi2_(2,VFloat(128,sistrip::invalid_)),
    mean_amplitude_(2,sistrip::invalid_),
    mean_tail_(2,sistrip::invalid_),
    mean_riseTime_(2,sistrip::invalid_),
    mean_timeConstant_(2,sistrip::invalid_),
    mean_turnOn_(2,sistrip::invalid_),
    mean_maximum_(2,sistrip::invalid_),
    mean_undershoot_(2,sistrip::invalid_),
    mean_baseline_(2,sistrip::invalid_),
    mean_smearing_(2,sistrip::invalid_),
    mean_chi2_(2,sistrip::invalid_),
    min_amplitude_(2,sistrip::invalid_),
    min_tail_(2,sistrip::invalid_),
    min_riseTime_(2,sistrip::invalid_),
    min_timeConstant_(2,sistrip::invalid_),
    min_turnOn_(2,sistrip::invalid_),
    min_maximum_(2,sistrip::invalid_),
    min_undershoot_(2,sistrip::invalid_),
    min_baseline_(2,sistrip::invalid_),
    min_smearing_(2,sistrip::invalid_),
    min_chi2_(2,sistrip::invalid_),
    max_amplitude_(2,sistrip::invalid_),
    max_tail_(2,sistrip::invalid_),
    max_riseTime_(2,sistrip::invalid_),
    max_timeConstant_(2,sistrip::invalid_),
    max_turnOn_(2,sistrip::invalid_),
    max_maximum_(2,sistrip::invalid_),
    max_undershoot_(2,sistrip::invalid_),
    max_baseline_(2,sistrip::invalid_),
    max_smearing_(2,sistrip::invalid_),
    max_chi2_(2,sistrip::invalid_),
    spread_amplitude_(2,sistrip::invalid_),
    spread_tail_(2,sistrip::invalid_),
    spread_riseTime_(2,sistrip::invalid_),
    spread_timeConstant_(2,sistrip::invalid_),
    spread_turnOn_(2,sistrip::invalid_),
    spread_maximum_(2,sistrip::invalid_),
    spread_undershoot_(2,sistrip::invalid_),
    spread_baseline_(2,sistrip::invalid_),
    spread_smearing_(2,sistrip::invalid_),
    spread_chi2_(2,sistrip::invalid_),
    deconv_(deconv),
    calchan_(calchan),
    isScan_(false)
{;}

// ----------------------------------------------------------------------------
// 
CalibrationAnalysis::CalibrationAnalysis(const bool& deconv, int calchan) 
  : CommissioningAnalysis("CalibrationAnalysis"),
    amplitude_(2,VFloat(128,sistrip::invalid_)),
    tail_(2,VFloat(128,sistrip::invalid_)),
    riseTime_(2,VFloat(128,sistrip::invalid_)),
    timeConstant_(2,VFloat(128,sistrip::invalid_)),
    turnOn_(2,VFloat(128,sistrip::invalid_)),
    maximum_(2,VFloat(128,sistrip::invalid_)),
    undershoot_(2,VFloat(128,sistrip::invalid_)),
    baseline_(2,VFloat(128,sistrip::invalid_)),
    smearing_(2,VFloat(128,sistrip::invalid_)),
    chi2_(2,VFloat(128,sistrip::invalid_)),
    mean_amplitude_(2,sistrip::invalid_),
    mean_tail_(2,sistrip::invalid_),
    mean_riseTime_(2,sistrip::invalid_),
    mean_timeConstant_(2,sistrip::invalid_),
    mean_turnOn_(2,sistrip::invalid_),
    mean_maximum_(2,sistrip::invalid_),
    mean_undershoot_(2,sistrip::invalid_),
    mean_baseline_(2,sistrip::invalid_),
    mean_smearing_(2,sistrip::invalid_),
    mean_chi2_(2,sistrip::invalid_),
    min_amplitude_(2,sistrip::invalid_),
    min_tail_(2,sistrip::invalid_),
    min_riseTime_(2,sistrip::invalid_),
    min_timeConstant_(2,sistrip::invalid_),
    min_turnOn_(2,sistrip::invalid_),
    min_maximum_(2,sistrip::invalid_),
    min_undershoot_(2,sistrip::invalid_),
    min_baseline_(2,sistrip::invalid_),
    min_smearing_(2,sistrip::invalid_),
    min_chi2_(2,sistrip::invalid_),
    max_amplitude_(2,sistrip::invalid_),
    max_tail_(2,sistrip::invalid_),
    max_riseTime_(2,sistrip::invalid_),
    max_timeConstant_(2,sistrip::invalid_),
    max_turnOn_(2,sistrip::invalid_),
    max_maximum_(2,sistrip::invalid_),
    max_undershoot_(2,sistrip::invalid_),
    max_baseline_(2,sistrip::invalid_),
    max_smearing_(2,sistrip::invalid_),
    max_chi2_(2,sistrip::invalid_),
    spread_amplitude_(2,sistrip::invalid_),
    spread_tail_(2,sistrip::invalid_),
    spread_riseTime_(2,sistrip::invalid_),
    spread_timeConstant_(2,sistrip::invalid_),
    spread_turnOn_(2,sistrip::invalid_),
    spread_maximum_(2,sistrip::invalid_),
    spread_undershoot_(2,sistrip::invalid_),
    spread_baseline_(2,sistrip::invalid_),
    spread_smearing_(2,sistrip::invalid_),
    spread_chi2_(2,sistrip::invalid_),
    deconv_(deconv),
    calchan_(calchan),
    isScan_(false)
{;}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::reset() {
  amplitude_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  tail_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  riseTime_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  timeConstant_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  turnOn_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  maximum_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  undershoot_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  baseline_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  smearing_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  chi2_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  mean_amplitude_ = VFloat(2,sistrip::invalid_);
  mean_tail_ = VFloat(2,sistrip::invalid_);
  mean_riseTime_ = VFloat(2,sistrip::invalid_);
  mean_timeConstant_ = VFloat(2,sistrip::invalid_);
  mean_turnOn_ = VFloat(2,sistrip::invalid_);
  mean_maximum_ = VFloat(2,sistrip::invalid_);
  mean_undershoot_ = VFloat(2,sistrip::invalid_);
  mean_baseline_ = VFloat(2,sistrip::invalid_);
  mean_smearing_ = VFloat(2,sistrip::invalid_);
  mean_chi2_ = VFloat(2,sistrip::invalid_);
  min_amplitude_ = VFloat(2,sistrip::invalid_);
  min_tail_ = VFloat(2,sistrip::invalid_);
  min_riseTime_ = VFloat(2,sistrip::invalid_);
  min_timeConstant_ = VFloat(2,sistrip::invalid_);
  min_turnOn_ = VFloat(2,sistrip::invalid_);
  min_maximum_ = VFloat(2,sistrip::invalid_);
  min_undershoot_ = VFloat(2,sistrip::invalid_);
  min_baseline_ = VFloat(2,sistrip::invalid_);
  min_smearing_ = VFloat(2,sistrip::invalid_);
  min_chi2_ = VFloat(2,sistrip::invalid_);
  max_amplitude_ = VFloat(2,sistrip::invalid_);
  max_tail_ = VFloat(2,sistrip::invalid_);
  max_riseTime_ = VFloat(2,sistrip::invalid_);
  max_timeConstant_ = VFloat(2,sistrip::invalid_);
  max_turnOn_ = VFloat(2,sistrip::invalid_);
  max_maximum_ = VFloat(2,sistrip::invalid_);
  max_undershoot_ = VFloat(2,sistrip::invalid_);
  max_baseline_ = VFloat(2,sistrip::invalid_);
  max_smearing_ = VFloat(2,sistrip::invalid_);
  max_chi2_ = VFloat(2,sistrip::invalid_);
  spread_amplitude_ = VFloat(2,sistrip::invalid_);
  spread_tail_ = VFloat(2,sistrip::invalid_);
  spread_riseTime_ = VFloat(2,sistrip::invalid_);
  spread_timeConstant_ = VFloat(2,sistrip::invalid_);
  spread_turnOn_ = VFloat(2,sistrip::invalid_);
  spread_maximum_ = VFloat(2,sistrip::invalid_);
  spread_undershoot_ = VFloat(2,sistrip::invalid_);
  spread_baseline_ = VFloat(2,sistrip::invalid_);
  spread_smearing_ = VFloat(2,sistrip::invalid_);
  spread_chi2_ = VFloat(2,sistrip::invalid_);
}

// ----------------------------------------------------------------------------
// 
void CalibrationAnalysis::print( std::stringstream& ss, uint32_t iapv ) { 
  header( ss );
  ss << " Monitorables for APV number     : " << iapv;
  if ( iapv == 0 ) { ss << " (first of pair)"; }
  else if ( iapv == 1 ) { ss << " (second of pair)"; }
  ss << std::endl;
  ss << " Mean Amplitude of the pulse : " << mean_amplitude_[iapv] << std::endl
     << " Mean Tail amplitude after 150ns : " << mean_tail_[iapv] << std::endl
     << " Mean Rise time : " << mean_riseTime_[iapv] << std::endl
     << " Mean Time constant : " << mean_timeConstant_[iapv] << std::endl
     << " Mean Turn on time  : " << mean_turnOn_[iapv] << std::endl
     << " Mean peak time : " << mean_maximum_[iapv] << std::endl
     << " Mean undershoot amplitude : " << mean_undershoot_[iapv] << std::endl
     << " Mean baseline amplitude : " << mean_baseline_[iapv] << std::endl
     << " Mean Smearing parameter : " << mean_smearing_[iapv] << std::endl
     << " Mean Chi2 of the fit : " << mean_chi2_[iapv] << std::endl;
  if(deconvMode()) {
    ss << "Data obtained in deconvolution mode." << std::endl;
  } else {
    ss << "Data obtained in peak mode." << std::endl;
  }
}

