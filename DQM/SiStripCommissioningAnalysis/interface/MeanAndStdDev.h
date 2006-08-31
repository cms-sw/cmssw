#ifndef DQM_SiStripCommissioningAnalysis_MeanAndStdDev_H
#define DQM_SiStripCommissioningAnalysis_MeanAndStdDev_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <vector>

/** */
class MeanAndStdDev {
  
 public:
  
  MeanAndStdDev();
  ~MeanAndStdDev() {;}

  class Params {
  public:
    float mean_;
    float rms_;
    float median_;
    Params() :
      mean_(sistrip::invalid_), 
      rms_(sistrip::invalid_), 
      median_(sistrip::invalid_) {;}
    ~Params() {;}
  };
  
  /** */
  void add( const float& value,
	    const float& error );
  
  void fit( Params& fit_params );
  
 private:
  
  float s_;
  float x_;
  float xx_;
  std::vector<float> vec_;
  
};

#endif // DQM_SiStripCommissioningAnalysis_MeanAndStdDev_H

