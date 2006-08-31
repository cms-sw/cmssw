#ifndef DQM_SiStripCommissioningAnalysis_LinearFit_H
#define DQM_SiStripCommissioningAnalysis_LinearFit_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <vector>

/** */
class LinearFit {
  
 public:
  
  LinearFit();
  ~LinearFit() {;}

  class Params {
  public:
    uint16_t n_;
    float a_;
    float b_;
    float erra_;
    float errb_;
    Params() :
      n_(sistrip::invalid_), 
      a_(sistrip::invalid_), 
      b_(sistrip::invalid_), 
      erra_(sistrip::invalid_), 
      errb_(sistrip::invalid_) {;}
    ~Params() {;}
  };
  
  /** */
  void add( const float& value_x,
	    const float& value_y, 
	    const float& error_y );
  
  void fit( Params& fit_params );
  
 private:
  
  std::vector<float> x_;
  std::vector<float> y_;
  std::vector<float> e_;
  float ss_;
  float sx_;
  float sy_;
  
};

#endif // DQM_SiStripCommissioningAnalysis_LinearFit_H

