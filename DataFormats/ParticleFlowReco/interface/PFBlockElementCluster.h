#ifndef __PFBlockElementCluster__
#define __PFBlockElementCluster__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

namespace reco {
  
  class PFBlockElementCluster : public PFBlockElement {
  public:
    PFBlockElementCluster() :
      PFBlockElement( NONE ) {} 
    
    PFBlockElementCluster(const PFClusterRef& ref, 
			  PFBlockElement::Type type) 
      : 
      PFBlockElement(type),
      clusterRef_( ref ) {}
    
    PFBlockElement* clone() const { return new PFBlockElementCluster(*this); }

    PFClusterRef  clusterRef() const {return clusterRef_;}

    void Dump(std::ostream& out = std::cout, 
	      const char* tab = " " ) const;
  
  private:
    PFClusterRef  clusterRef_;
  };
}

#endif

