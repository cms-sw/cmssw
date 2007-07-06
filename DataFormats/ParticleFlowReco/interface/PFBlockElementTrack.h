#ifndef __PFBlockElementTrack__
#define __PFBlockElementTrack__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  
  /// \brief Track Element.
  /// 
  /// this class contains a reference to a PFRecTrack 
  class PFBlockElementTrack : public PFBlockElement {
  public:
    PFBlockElementTrack() {} 

    PFBlockElementTrack(const PFRecTrackRef& ref );

    PFBlockElement* clone() const { return new PFBlockElementTrack(*this); }
    
    void Dump(std::ostream& out = std::cout, 
	      const char* tab = " " ) const;
    
    /// \return reference to the corresponding PFRecTrack
    PFRecTrackRef trackRefPF() const {
      return trackRefPF_;
    }
    
    /// \return reference to the corresponding Track
    reco::TrackRef trackRef() const {
      return trackRef_;
    }
    
  private:

    /// reference to the corresponding track (transient)
    PFRecTrackRef  trackRefPF_;

    /// reference to the corresponding track 
    reco::TrackRef trackRef_;
  };
}

#endif

