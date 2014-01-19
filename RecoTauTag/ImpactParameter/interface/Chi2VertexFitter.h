#ifndef RecoTauTag_ImpactParameter_Chi2VertexFitter_h
#define RecoTauTag_ImpactParameter_Chi2VertexFitter_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/FCNBase.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"

namespace tauImpactParameter {

class  Chi2VertexFitter : public TrackHelixVertexFitter {
 public:
  Chi2VertexFitter(const std::vector<TrackParticle>& particles, const TVector3& vguess, double nsigma=4.0)
    : TrackHelixVertexFitter(particles,vguess),
      nsigma_(nsigma)
  {};
  virtual ~Chi2VertexFitter(){};

  virtual bool fit();

 private:   
  double nsigma_;
};

}
#endif


