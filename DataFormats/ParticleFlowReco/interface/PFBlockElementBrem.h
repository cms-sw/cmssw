#ifndef __PFBlockElementBrem__
#define __PFBlockElementBrem__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBrem.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  
  /// \brief Track Element.
  /// 
  /// this class contains a reference to a PFRecTrack 
  class PFBlockElementBrem : public PFBlockElement {
  public:

    PFBlockElementBrem() {} 

    PFBlockElementBrem(const GsfPFRecTrackRef& gsfref, const double DeltaP, const double SigmaDeltaP, const unsigned int indTrajPoint);

      
    PFBlockElement* clone() const override { return new PFBlockElementBrem(*this); }
    void Dump(std::ostream& out = std::cout, 
              const char* tab = " " ) const override;
    

    const GsfPFRecTrackRef& GsftrackRefPF() const {
      return GsftrackRefPF_;
    }
    
    /// \return reference to the corresponding Track
    const reco::GsfTrackRef& GsftrackRef() const {
      return GsftrackRef_;
    }
    
    const PFRecTrack & trackPF() const    
      { return ((*GsftrackRefPF()).PFRecBrem()[(indPoint_-2)]);}     
    

    double DeltaP() const {return deltaP_;}
    double SigmaDeltaP() const {return sigmadeltaP_;}
    unsigned int indTrajPoint() const {return indPoint_;}

    /// \return position at ECAL entrance
    const math::XYZPointF& positionAtECALEntrance() const {
      return positionAtECALEntrance_;
    }
      

  private:
    
    /// reference to the corresponding track (transient)
    GsfPFRecTrackRef  GsftrackRefPF_;
   
    /// reference to the corresponding track 
    reco::GsfTrackRef GsftrackRef_;
     
    double deltaP_;
    double sigmadeltaP_;
    unsigned int indPoint_;
    math::XYZPointF        positionAtECALEntrance_;
 
  };
}

#endif

