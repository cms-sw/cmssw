#ifndef __PFBlockElementBrem__
#define __PFBlockElementBrem__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  
  /// \brief Track Element.
  /// 
  /// this class contains a reference to a PFRecTrack 
  class PFBlockElementBrem : public PFBlockElementTrack {
  public:

    PFBlockElementBrem() {} 

    PFBlockElementBrem(const PFRecTrackRef& ref , TrackType tracktype):
      PFBlockElementTrack( ref, tracktype  ),
      BremtrackRefPF_( ref ), 
      BremtrackRef_( ref->trackRef() ),
      deltaP_(DeltaP()),
      sigmadeltaP_(SigmaDeltaP()),
      indPoint_(indTrajPoint()){}

      
    PFBlockElement* clone() const { return new PFBlockElementBrem(*this); }
    
    
    /// \return reference to the corresponding PFRecTrack
    PFRecTrackRef trackRefPF() const {
      return BremtrackRefPF_;
    }
    
    /// \return reference to the corresponding Track
    reco::TrackRef trackRef() const {
      return BremtrackRef_;
    }

    double DeltaP(){return deltaP_;}
    double SigmaDeltaP(){return sigmadeltaP_;}
    uint indTrajPoint() {return indPoint_;}
  private:
    
    /// reference to the corresponding track (transient)
    PFRecTrackRef  BremtrackRefPF_;
    
    /// reference to the corresponding track 
    reco::TrackRef BremtrackRef_;
     
    double deltaP_;
    double sigmadeltaP_;
    uint indPoint_;

 
  };
}

#endif

