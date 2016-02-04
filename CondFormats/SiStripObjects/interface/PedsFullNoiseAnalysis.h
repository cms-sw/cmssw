#ifndef CondFormats_SiStripObjects_PedsFullNoiseAnalysis_H
#define CondFormats_SiStripObjects_PedsFullNoiseAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

/** 
    @class PedsFullNoiseAnalysis
    @author M. Wingham, R.Bainbridge
    @brief Histogram-based analysis for pedestal run.
*/
class PedsFullNoiseAnalysis : public CommissioningAnalysis {
  
 public:

	// ---------- con(de)structors ----------

    PedsFullNoiseAnalysis( const uint32_t& key );

    PedsFullNoiseAnalysis();

    virtual ~PedsFullNoiseAnalysis() {;}

    friend class PedestalsAlgorithm;
    friend class PedsFullNoiseAlgorithm;

    // ---------- public interface ----------

  	/** Identifies if analysis is valid or not. */
  	bool isValid() const;
  
  	// Pedestal, noise and raw noise (128-strip vector per APV)
	inline const VVFloat& peds() const;
    inline const VVFloat& noise() const;
    inline const VVFloat& raw() const;
    // KS Probability for each strip
    inline const VVFloat& ksProb() const;
    // KS Probability for each strip
    inline const VVFloat& chi2Prob() const;
    // Noise value calculated by a gaussian fit instead of RMS.
	inline const VVFloat& noiseGaus() const;
	// Noise value calculated by a gaussian fit instead of RMS.
	inline const VVFloat& noiseBin84() const;
    // Noise value calculated by RMS of ADC values.
    inline const VVFloat& noiseRMS() const;
    // The significance of noise of each strip compared to the apv
	inline const VVFloat& noiseSignif() const;
    
	// Dead and noisy strips (vector per APV)
    inline const VVInt& dead() const; 
    inline const VVInt& noisy() const;

    // Mean and rms spread (value per APV)
    inline const VFloat& pedsMean() const;
    inline const VFloat& pedsSpread() const;
    inline const VFloat& noiseMean() const;
    inline const VFloat& noiseSpread() const;
    inline const VFloat& rawMean() const;
    inline const VFloat& rawSpread() const;

    // Max and min values (value per APV)
    inline const VFloat& pedsMax() const;
    inline const VFloat& pedsMin() const; 
    inline const VFloat& noiseMax() const;
    inline const VFloat& noiseMin() const;
    inline const VFloat& rawMax() const;
    inline const VFloat& rawMin() const;

    // ---------- misc ----------

    /** Prints analysis results. */
    void print( std::stringstream&, uint32_t apv_number = 0 );

    /** Overrides base method. */
    void summary( std::stringstream& ) const;

    /** Resets analysis member data. */
    void reset();

    // ---------- private member data ----------

  private:
	
    // VVFloats means: 1 vector per APV, 1 value per strip.

    /** Peds values. */
    VVFloat peds_;

    /** Noise values. */
    VVFloat noise_;

    /** Raw noise values. */
    VVFloat raw_;


    VVFloat ksProb_;
    VVFloat chi2Prob_;
    VVFloat noiseGaus_;
	VVFloat noiseBin84_;
    VVFloat noiseRMS_;
    VVFloat noiseSignif_;
    // VVInts means: 1 vector per APV, values are strip numbers.

    /** Dead strips. */
    VVInt dead_; 

    /** Noisy strips. */
    VVInt noisy_;

    // VFloat: 1 value per APV

    /** Mean peds value. */
    VFloat pedsMean_;

    /** Rms spread in peds. */
    VFloat pedsSpread_;

    /** Mean noise value. */
    VFloat noiseMean_;

    /** Rms spread in noise. */
    VFloat noiseSpread_;

    /** Mean raw noise value. */
    VFloat rawMean_;

    /** Rms spread in raw noise. */
    VFloat rawSpread_;

    /** Max peds value. */
    VFloat pedsMax_;

    /** Min peds value. */
    VFloat pedsMin_; 

    /** Max noise value. */
    VFloat noiseMax_;

    /** Min noise value. */
    VFloat noiseMin_;

    /** Max raw noise value. */
    VFloat rawMax_;

    /** Min raw noise value. */
    VFloat rawMin_;

    // true if legacy histogram naming is used
    bool legacy_;
};

// ---------- Inline methods ----------

const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::peds() const { return peds_; }
const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::noise() const { return noise_; }
const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::raw() const { return raw_; }
const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::ksProb() const { return ksProb_; }
const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::chi2Prob() const { return chi2Prob_; }
const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::noiseGaus() const { return noiseGaus_; }
const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::noiseBin84() const { return noiseBin84_; }
const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::noiseRMS() const { return noiseRMS_; }
const PedsFullNoiseAnalysis::VVFloat& PedsFullNoiseAnalysis::noiseSignif() const { return noiseSignif_; }

const PedsFullNoiseAnalysis::VVInt& PedsFullNoiseAnalysis::dead() const { return dead_; } 
const PedsFullNoiseAnalysis::VVInt& PedsFullNoiseAnalysis::noisy() const { return noisy_; }

const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::pedsMean() const { return pedsMean_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::pedsSpread() const { return pedsSpread_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::noiseMean() const { return noiseMean_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::noiseSpread() const { return noiseSpread_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::rawMean() const { return rawMean_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::rawSpread() const { return rawSpread_; }

const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::pedsMax() const { return pedsMax_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::pedsMin() const { return pedsMin_; } 
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::noiseMax() const { return noiseMax_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::noiseMin() const { return noiseMin_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::rawMax() const { return rawMax_; }
const PedsFullNoiseAnalysis::VFloat& PedsFullNoiseAnalysis::rawMin() const { return rawMin_; }
#endif // CondFormats_SiStripObjects_PedsFullNoiseAnalysis_H
