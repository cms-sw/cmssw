#include "Fireworks/ParticleFlow/interface/FWLegoEvePFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

FWLegoEvePFCandidate::FWLegoEvePFCandidate(const reco::PFCandidate& iData) {

  const unsigned int nLineSegments = 20;
  
  float circleScalingFactor = 50;
  float lineScalingFactor = 1;
  
  const double jetRadius = iData.pt()/circleScalingFactor;
  
  for ( unsigned int iphi = 0; iphi < nLineSegments; ++iphi ) {
    AddLine(iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*iphi),
	    iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*iphi),
	    0.1,
	    iData.eta()+jetRadius*cos(2*M_PI/nLineSegments*(iphi+1)),
	    iData.phi()+jetRadius*sin(2*M_PI/nLineSegments*(iphi+1)),
	    0.1);
  }
  

  AddLine(iData.eta(),iData.phi(), 0.1, 
	  iData.eta(),iData.phi(), 0.1 + iData.et()/lineScalingFactor );

}
