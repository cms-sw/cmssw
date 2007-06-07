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

  // ---------- Con(de)structors and typedefs ----------

  PedestalsAnalysis( const uint32_t& key );
  PedestalsAnalysis();
  virtual ~PedestalsAnalysis() {;}

  // ---------- Access to analysis and histos ----------
  
  // Pedestal, noise and raw noise vectors (per APV)
  inline const VVFloat& peds() const;
  inline const VVFloat& noise() const;
  inline const VVFloat& raw() const;

  // Dead and noisy strips (per APV)
  inline const VVInt& dead() const; 
  inline const VVInt& noisy() const;

  // Mean and rms spread
  inline const VFloat& pedsMean() const;
  inline const VFloat& pedsSpread() const;
  inline const VFloat& noiseMean() const;
  inline const VFloat& noiseSpread() const;
  inline const VFloat& rawMean() const;
  inline const VFloat& rawSpread() const;

  // Max and min values
  inline const VFloat& pedsMax() const;
  inline const VFloat& pedsMin() const; 
  inline const VFloat& noiseMax() const;
  inline const VFloat& noiseMin() const;
  inline const VFloat& rawMax() const;
  inline const VFloat& rawMin() const;

  inline const Histo& hPeds() const;
  inline const Histo& hNoise() const;

  // ---------- Utility methods ----------
  
  bool isValid();

  void print( std::stringstream&, uint32_t apv_number = 0 );

 private:
  
  void reset();
  void extract( const std::vector<TH1*>& );
  void analyse();
  
 private:
  
  /** Peds values (1 value per strip, 1 vector per APV) */
  VVFloat peds_;
  /** Noise values (1 value per strip, 1 vector per APV) */
  VVFloat noise_;
  /** Raw noise values (1 value per strip, 1 vector per APV) */
  VVFloat raw_;

  /** Dead strips (values are strip numbers, 1 vector per APV) */
  VVInt dead_; 
  /** Noisy strips (values are strip numbers, 1 vector per APV) */
  VVInt noisy_;

  /** Mean peds value (1 value per APV) */
  VFloat pedsMean_;
  /** Rms spread in peds (1 value per APV) */
  VFloat pedsSpread_;
  /** Mean noise value (1 value per APV) */
  VFloat noiseMean_;
  /** Rms spread in noise (1 value per APV) */
  VFloat noiseSpread_;
  /** Mean raw noise value (1 value per APV) */
  VFloat rawMean_;
  /** Rms spread in raw noise (1 value per APV) */
  VFloat rawSpread_;

  /** Max peds value (1 value per APV) */
  VFloat pedsMax_;
  /** Min peds value (1 value per APV) */
  VFloat pedsMin_; 
  /** Max noise value (1 value per APV) */
  VFloat noiseMax_;
  /** Min noise value (1 value per APV) */
  VFloat noiseMin_;
  /** Max raw noise value (1 value per APV) */
  VFloat rawMax_;
  /** Min raw noise value (1 value per APV) */
  VFloat rawMin_;
  
  /** Pedestals and raw noise */
  Histo hPeds_;
  /** Residuals and noise */
  Histo hNoise_;
  
};

const PedestalsAnalysis::VVFloat& PedestalsAnalysis::peds() const { return peds_; }
const PedestalsAnalysis::VVFloat& PedestalsAnalysis::noise() const { return noise_; }
const PedestalsAnalysis::VVFloat& PedestalsAnalysis::raw() const { return raw_; }

const PedestalsAnalysis::VVInt& PedestalsAnalysis::dead() const { return dead_; } 
const PedestalsAnalysis::VVInt& PedestalsAnalysis::noisy() const { return noisy_; }

const PedestalsAnalysis::VFloat& PedestalsAnalysis::pedsMean() const { return pedsMean_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::pedsSpread() const { return pedsSpread_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::noiseMean() const { return noiseMean_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::noiseSpread() const { return noiseSpread_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::rawMean() const { return rawMean_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::rawSpread() const { return rawSpread_; }

const PedestalsAnalysis::VFloat& PedestalsAnalysis::pedsMax() const { return pedsMax_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::pedsMin() const { return pedsMin_; } 
const PedestalsAnalysis::VFloat& PedestalsAnalysis::noiseMax() const { return noiseMax_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::noiseMin() const { return noiseMin_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::rawMax() const { return rawMax_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::rawMin() const { return rawMin_; }

const PedestalsAnalysis::Histo& PedestalsAnalysis::hPeds() const { return hPeds_; }
const PedestalsAnalysis::Histo& PedestalsAnalysis::hNoise() const { return hNoise_; }

#endif // DQM_SiStripCommissioningAnalysis_PedestalsAnalysis_H
