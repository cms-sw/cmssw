#ifndef CondFormats_SiStripObjects_LatencyAnalysis_H
#define CondFormats_SiStripObjects_LatencyAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;
class TF1;

/**
   @class LatencyAnalysis
   @author C. Delaere
   @brief Analysis for latency run
*/

class LatencyAnalysis : public CommissioningAnalysis {
  
 public:
  
  LatencyAnalysis( const uint32_t& key );
  LatencyAnalysis();
  virtual ~LatencyAnalysis() {;}
  
  inline const float& maximum() const { return max_; }
  inline const float& error() const { return error_; }
  
  inline const Histo& histo() const;
  void print( std::stringstream&, uint32_t not_used = 0 );
  
 private:
  
  void reset();
  virtual void extract( const std::vector<TH1*>& );
  void analyse();
  void pruneProfile(TProfile* profile) const;
  void correctBinning(TProfile* prof) const;
  float limit(float SoNcut) const;
  float correctMeasurement(float mean, float SoNcut=3.) const;
  void correctProfile(TProfile* profile, float SoNcut=3.) const;
  
 private:
  
  /** Delay corresponding to the maximum of the pulse shape */
  float max_;
  /** Error on the position ( from the fit) */
  float error_;
  /** pulse shape*/
  Histo histo_;
  /** Fitter in deconvolution mode */
  TF1* peak_fitter_;
  
};

#endif // CondFormats_SiStripObjects_LatencyAnalysis_H



