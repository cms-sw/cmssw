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
    InverseCrosstalkMatrix(unsigned N, float x);
    float operator()(unsigned i, unsigned j) const;
    
  private:
    float element(unsigned, unsigned) const;
    const unsigned N;
    const double sq, lambdaP, lambdaM, denominator;
    
  public:
    static std::vector<stats_t<float> > unfold(const SiStripCluster& q, float x);
    
  };
}
#endif
