/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#ifndef Chi2VertexFitter_h
#define Chi2VertexFitter_h

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/FCNBase.h"
#include "RecoTauTag/ImpactParameter/interface/TrackHelixVertexFitter.h"

class  Chi2VertexFitter : public TrackHelixVertexFitter {
 public:
  Chi2VertexFitter(std::vector<TrackParticle> &particles,TVector3 vguess,double nsigma_=4.0):TrackHelixVertexFitter(particles,vguess),nsigma(nsigma_){};
  virtual ~Chi2VertexFitter(){};

  virtual bool Fit();

 private:   
  double nsigma;
};
#endif


