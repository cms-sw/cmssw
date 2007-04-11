#ifndef __PFBlockElementTrack__
#define __PFBlockElementTrack__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"

namespace reco {
  
  class PFBlockElementTrack : public PFBlockElement {
  public:
    PFBlockElementTrack() {} 

    PFBlockElementTrack(const PFRecTrackRef& ref ) : 
      PFBlockElement( TRACK ),
      trackRef_( ref ) {}

    PFBlockElement* clone() const { return new PFBlockElementTrack(*this); }
    
    void Dump(std::ostream& out = std::cout, 
	      const char* tab = " " ) const;
    
    PFRecTrackRef  trackRef() const {
      return trackRef_;
    }
    
  private:
    PFRecTrackRef  trackRef_;
  };
}

#endif

