#ifndef DataFormats_ParticleFlowReco_PFBrem_h
#define DataFormats_ParticleFlowReco_PFBrem_h

#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"

namespace reco {

 
  class PFBrem : public PFRecTrack {

  public:
 
    PFBrem(){}
    PFBrem(double DP,
	   double SigmaDP,
	   uint PointInd): 
      deltaP_(DP),
      sigmadeltaP_(SigmaDP),
      indPoint_(PointInd) {}
      
      
      double DeltaP(){return deltaP_;}
      double SigmaDeltaP(){return sigmadeltaP_;}
      uint indTrajPoint() {return indPoint_;}
  private:
      
      double deltaP_;
      double sigmadeltaP_;
      uint indPoint_;
  };
  
  
}

#endif
