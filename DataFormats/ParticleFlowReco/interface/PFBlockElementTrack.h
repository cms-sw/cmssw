#ifndef __PFBlockElementTrack__
#define __PFBlockElementTrack__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"

namespace reco {
  
  /// \brief Track Element.
  /// 
  /// this class contains a reference to a PFRecTrack 
  class PFBlockElementTrack : public PFBlockElement {
  public:
    PFBlockElementTrack() {} 

    PFBlockElementTrack(const PFRecTrackRef& ref ) : 
      PFBlockElement( TRACK ),
      trackRef_( ref ) {}

    PFBlockElement* clone() const { return new PFBlockElementTrack(*this); }
    
    void Dump(std::ostream& out = std::cout, 
	      const char* tab = " " ) const;
    
    /// \return reference to the corresponding track
    PFRecTrackRef  trackRef() const {
      return trackRef_;
    }
    
  private:
    /// reference to the corresponding track
    PFRecTrackRef  trackRef_;
  };
}

#endif

