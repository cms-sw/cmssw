#ifndef __PFBlockElementGsfTrack__
#define __PFBlockElementGsfTrack__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  
  /// \brief Track Element.
  /// 
  /// this class contains a reference to a PFRecTrack 
  class PFBlockElementGsfTrack : public PFBlockElementTrack {
  public:

    PFBlockElementGsfTrack() {} 

    PFBlockElementGsfTrack(const PFRecTrackRef& Gsfref, const PFRecTrackRef& Kfref, TrackType tracktype);

    PFBlockElement* clone() const { return new PFBlockElementGsfTrack(*this); }
    
    void Dump(std::ostream& out = std::cout, 
              const char* tab = " " ) const;

    
    /// \return reference to the corresponding PFRecTrack
    PFRecTrackRef GsftrackRefPF() const {
      return GsftrackRefPF_;
    }
    
    /// \return reference to the corresponding Track
    reco::TrackRef GsftrackRef() const {
      return GsftrackRef_;
    }

  
    const math::XYZTLorentzVector& Pin() const    { return Pin_; }
    const math::XYZTLorentzVector& Pout() const    { return Pout_; }
  private:
    
    /// reference to the corresponding GSF track (transient)
    PFRecTrackRef  GsftrackRefPF_;
    
    /// reference to the corresponding GSF track 
    reco::TrackRef GsftrackRef_;
      
    
    /// The CorrespondingKFTrackRef is needeed. 
    math::XYZTLorentzVector Pin_;
    math::XYZTLorentzVector Pout_;

  };
}

#endif

