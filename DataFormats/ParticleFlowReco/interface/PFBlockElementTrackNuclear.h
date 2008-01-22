#ifndef __PFBlockElementTrackNuclear__
#define __PFBlockElementTrackNuclear__

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFNuclearInteraction.h"

namespace reco {
  
  /// \brief Track Element.
  /// 
  /// this class contains a reference to a PFRecTrack 
  class PFBlockElementTrackNuclear : public PFBlockElementTrack {
  public:
    PFBlockElementTrackNuclear() {} 

    PFBlockElementTrackNuclear(const PFRecTrackRef& ref, const PFNuclearInteractionRef& niRef_ ) : 
           PFBlockElementTrack( ref , TRACKNUCL), pfNuclInterRef_( niRef_ ),
           nuclInterRef_(niRef_->nuclInterRef()) {}

    PFBlockElement* clone() const { return new PFBlockElementTrackNuclear(*this); }
    
  private:

    /// reference to the corresponding pf nuclear interaction
    PFNuclearInteractionRef  pfNuclInterRef_;

    /// reference to the corresponding nuclear interaction
    NuclearInteractionRef    nuclInterRef_;
  };
}

#endif

