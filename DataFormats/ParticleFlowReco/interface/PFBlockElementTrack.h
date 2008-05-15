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

    PFBlockElementTrack(const PFRecTrackRef& ref , TrackType trackType=DEFAULT );

    PFBlockElement* clone() const { return new PFBlockElementTrack(*this); }
    
    void Dump(std::ostream& out = std::cout, 
              const char* tab = " " ) const;

    /// \return tracktype
    virtual TrackType trackType() const { return trackType_; }
    
    /// \return reference to the corresponding PFRecTrack
    PFRecTrackRef trackRefPF() const {
      return trackRefPF_;
    }
    
    /// \return reference to the corresponding Track
    reco::TrackRef trackRef() const {
      return trackRef_;
    }

    bool isSecondary() const { return trackType_==T_FROM_NUCL || trackType_==T_FROM_GAMMACONV; }
    
  private:

    /// reference to the corresponding track (transient)
    PFRecTrackRef  trackRefPF_;

    /// reference to the corresponding track 
    reco::TrackRef trackRef_;

    TrackType     trackType_;
  };
}

#endif

