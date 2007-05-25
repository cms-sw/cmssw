#ifndef DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H
#define DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TH1;

/** 
    @class PedestalsAnalysis
    @author M. Wingham, R.Bainbridge
    @brief Histogram-based analysis for pedestal run.
*/
class PedestalsAnalysis : public CommissioningAnalysis {
  
 public:

  PedestalsAnalysis( const uint32_t& key );
  PedestalsAnalysis();
  virtual ~PedestalsAnalysis() {;}
  
  inline const VVFloats& peds() const;
  inline const VVFloats& noise() const;
  inline const VVInts& dead() const; 
  inline const VVInts& noisy() const;
  inline const VFloats& pedsMean() const;
  inline const VFloats& pedsSpread() const;
  inline const VFloats& noiseMean() const;
  inline const VFloats& noiseSpread() const;
  inline const VFloats& pedsMax() const;
  inline const VFloats& pedsMin() const; 
  inline const VFloats& noiseMax() const;
  inline const VFloats& noiseMin() const;

  inline const Histo& hPeds() const;
  inline const Histo& hNoise() const;
  
  bool isValid();

  void print( std::stringstream&, uint32_t apv_number = 0 );

 private:
  
  void reset();
  void extract( const std::vector<TH1*>& );
  void analyse();
  
 private:
  
  /** Peds values (1 value per strip, 1 vector per APV) */
  VVFloats peds_;
  /** Noise values (1 value per strip, 1 vector per APV) */
  VVFloats noise_;
  /** Dead strips (values are strip numbers, 1 vector per APV) */
  VVInts dead_; 
  /** Noisy strips (values are strip numbers, 1 vector per APV) */
  VVInts noisy_;
  /** Mean peds value (1 value per APV) */
  VFloats pedsMean_;
  /** Rms spread in peds (1 value per APV) */
  VFloats pedsSpread_;
  /** Mean noise value (1 value per APV) */
  VFloats noiseMean_;
  /** Rms spread in noise (1 value per APV) */
  VFloats noiseSpread_;
  /** Max peds value (1 value per APV) */
  VFloats pedsMax_;
  /** Min peds value (1 value per APV) */
  VFloats pedsMin_; 
  /** Max noise value (1 value per APV) */
  VFloats noiseMax_;
  /** Min noise value (1 value per APV) */
  VFloats noiseMin_;
  
  /** Pedestals and raw noise */
  Histo hPeds_;
  /** Residuals and noise */
  Histo hNoise_;
  
};

const PedestalsAnalysis::VVFloats& PedestalsAnalysis::peds() const { return peds_; }
const PedestalsAnalysis::VVFloats& PedestalsAnalysis::noise() const { return noise_; }
const PedestalsAnalysis::VVInts& PedestalsAnalysis::dead() const { return dead_; } 
const PedestalsAnalysis::VVInts& PedestalsAnalysis::noisy() const { return noisy_; }
const PedestalsAnalysis::VFloats& PedestalsAnalysis::pedsMean() const { return pedsMean_; }
const PedestalsAnalysis::VFloats& PedestalsAnalysis::pedsSpread() const { return pedsSpread_; }
const PedestalsAnalysis::VFloats& PedestalsAnalysis::noiseMean() const { return noiseMean_; }
const PedestalsAnalysis::VFloats& PedestalsAnalysis::noiseSpread() const { return noiseSpread_; }
const PedestalsAnalysis::VFloats& PedestalsAnalysis::pedsMax() const { return pedsMax_; }
const PedestalsAnalysis::VFloats& PedestalsAnalysis::pedsMin() const { return pedsMin_; } 
const PedestalsAnalysis::VFloats& PedestalsAnalysis::noiseMax() const { return noiseMax_; }
const PedestalsAnalysis::VFloats& PedestalsAnalysis::noiseMin() const { return noiseMin_; }
const PedestalsAnalysis::Histo& PedestalsAnalysis::hPeds() const { return hPeds_; }
const PedestalsAnalysis::Histo& PedestalsAnalysis::hNoise() const { return hNoise_; }

#endif // DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H
