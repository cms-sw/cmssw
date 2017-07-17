#ifndef SiStripRecHitConverter_InverseCrosstalkMatrix_h
#define SiStripRecHitConverter_InverseCrosstalkMatrix_h

#include <vector>
#include <cmath>
#include <stdint.h>
#include "RecoLocalTracker/SiStripRecHitConverter/interface/ErrorPropogationTypes.h"

class SiStripCluster;

namespace reco {

  class InverseCrosstalkMatrix {
  public:
    InverseCrosstalkMatrix(const unsigned N, const float x);
    float operator()(const unsigned i, const unsigned j) const;
    
  private:
    float element(const unsigned, const unsigned) const;
    const unsigned N;
    const double sq, lambdaP, lambdaM, denominator;
    
  public:
    static std::vector<stats_t<float> > unfold(const SiStripCluster& q, const float x);
    
  };
}
#endif
