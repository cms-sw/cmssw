#ifndef __PFBlockElementTrackNuclear__
#define __PFBlockElementTrackNuclear__

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"

namespace reco {
  
  /// \brief Track Element.
  /// 
  /// this class contains a reference to a PFRecTrack 
  class PFBlockElementTrackNuclear : public PFBlockElementTrack {
  public:
    PFBlockElementTrackNuclear() {} 

    PFBlockElementTrackNuclear(const PFRecTrackRef& ref, const PFNuclearInteractionRef& niRef_ , TrackType tracktype) : 
           PFBlockElementTrack( ref , tracktype), pfNuclInterRef_( niRef_ ){}

    PFBlockElement* clone() const { return new PFBlockElementTrackNuclear(*this); }

    PFNuclearInteractionRef nuclearRef() const { return pfNuclInterRef_; }
    
  private:

    /// reference to the corresponding pf nuclear interaction
    PFNuclearInteractionRef  pfNuclInterRef_;
  };
}

#endif

