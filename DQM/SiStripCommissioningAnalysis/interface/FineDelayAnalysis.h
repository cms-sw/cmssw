#ifndef DQM_SiStripCommissioningAnalysis_FineDelayAnalysis_H
#define DQM_SiStripCommissioningAnalysis_FineDelayAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>
#include "TF1.h"

class TProfile;

/**
   @class FineDelayAnalysis
   @author C. Delaere
   @brief Analysis for fine delay run
*/

class FineDelayAnalysis : public CommissioningAnalysis {
  
 public:
  
  FineDelayAnalysis( const uint32_t& key );
  FineDelayAnalysis();
  virtual ~FineDelayAnalysis() {;}
  
  inline const float& maximum() const { return max_; }
  inline const float& error() const { return error_; }
  
  inline const Histo& histo() const;
  void print( std::stringstream&, uint32_t not_used = 0 );
  
 private:
  
  void reset();
  virtual void extract( const std::vector<TH1*>& );
  void analyse();
  
 private:
  
  /** Delay corresponding to the maximum of the pulse shape */
  float max_;
  /** Error on the position ( from the fit) */
  float error_;
  /** pulse shape*/
  Histo histo_;
  /** Fitter in deconvolution mode */
  TF1* deconv_fitter_;
  
};

#endif // DQM_SiStripCommissioningAnalysis_FineDelayAnalysis_H



