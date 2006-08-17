#ifndef DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H
#define DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <vector>

class TProfile;

/** 
    @file : DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h
    @class : PedestalsAnalysis
    @author : M. Wingham
    @brief : Histogram-based analysis for pedestals and noise "monitorables".
*/
class PedestalsAnalysis : public CommissioningAnalysis {
  
 public:

  PedestalsAnalysis() {;}
  virtual ~PedestalsAnalysis() {;}
  
  /** Simple container class that holds various parameter values that
      are extracted from the pedestals histograms by the analysis. */
  class Monitorables : public CommissioningAnalysis::Monitorables {
  public:
    typedef std::vector<float> VFloats;
    typedef std::vector<VFloats> VVFloats;
    typedef std::vector<uint16_t> VInts;
    typedef std::vector<VInts> VVInts;
    VVFloats peds_;        // peds values (1 value per strip, 1 vector per APV) 
    VVFloats noise_;       // noise values (1 value per strip, 1 vector per APV)
    VVInts   dead_;        // dead strips (values are strip numbers, 1 vector per APV)
    VVInts   noisy_;       // noisy strips (values are strip numbers, 1 vector per APV)
    VFloats  pedsMean_;    // mean peds value (1 value per APV)
    VFloats  noiseMean_;   // mean noise value (1 value per APV)
    VFloats  pedsMedian_;  // median peds value (1 value per APV)
    VFloats  noiseMedian_; // median noise value (1 value per APV)
    VFloats  pedsSpread_;  // rms spread in peds (1 value per APV)
    VFloats  noiseSpread_; // rms spread in noise (1 value per APV)
    VFloats  pedsMax_;     // max peds value (1 value per APV)
    VFloats  noiseMax_;    // max noise value (1 value per APV)
    VFloats  pedsMin_;     // min peds value (1 value per APV)
    VFloats  noiseMin_;    // min noise value (1 value per APV)
    Monitorables() : 
      peds_(2,VFloats(128,sistrip::invalid_)), noise_(2,VFloats(128,sistrip::invalid_)), 
      dead_(2,VInts(0,sistrip::invalid_)), noisy_(2,VInts(0,sistrip::invalid_)),
      pedsMean_(2,sistrip::invalid_), noiseMean_(2,sistrip::invalid_), 
      pedsMedian_(2,sistrip::invalid_), noiseMedian_(2,sistrip::invalid_), 
      pedsSpread_(2,sistrip::invalid_), noiseSpread_(2,sistrip::invalid_), 
      pedsMax_(2,sistrip::invalid_), noiseMax_(2,sistrip::invalid_), 
      pedsMin_(2,sistrip::invalid_), noiseMin_(2,sistrip::invalid_) 
      {
	peds_.reserve(256); noise_.reserve(256); 
	dead_.reserve(256); noisy_.reserve(256);
      }
    void print( std::stringstream& );
  };
  
  /** Takes a vector containing one TH1F of pedestals (error bars - raw noise) and one of residuals(error bars - noise) and fills the monitorables vector with 2 vectors - the first of pedestals, the second of noise. */
  static void analysis( const std::vector<const TProfile*>& histos, 
			std::vector< std::vector<float> >& monitorables );
  
  /** Takes TProfile histos containing an APV tick mark, extracts
      various parameter values from the histogram, and fills the
      Monitorables object. */
  static void analysis( const std::vector<const TProfile*>&, Monitorables& );

};

#endif // DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H

