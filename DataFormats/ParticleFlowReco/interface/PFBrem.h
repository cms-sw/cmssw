#ifndef DataFormats_ParticleFlowReco_PFBrem_h
#define DataFormats_ParticleFlowReco_PFBrem_h

//COLIN: one must include the correct header when inheriting from a class
//#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

namespace reco {

  
  class PFBrem : public PFRecTrack {

  public:
 
    PFBrem(){}
    PFBrem(double DP,
	   double SigmaDP,
	   unsigned int PointInd): 
      deltaP_(DP),
      sigmadeltaP_(SigmaDP),
      indPoint_(PointInd) {}
      
      
      double DeltaP() const {return deltaP_;}
      double SigmaDeltaP() const {return sigmadeltaP_;}
      unsigned int indTrajPoint() const {return indPoint_;}

  private:
      
      double deltaP_;
      double sigmadeltaP_;
      unsigned int indPoint_;
  };
  
  
}

#endif
