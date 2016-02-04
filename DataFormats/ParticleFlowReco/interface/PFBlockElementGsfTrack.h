#ifndef __PFBlockElementGsfTrack__
#define __PFBlockElementGsfTrack__

#include <iostream>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  
  /// \brief Track Element.
  /// 
  /// this class contains a reference to a PFRecTrack 
  class PFBlockElementGsfTrack : public PFBlockElement {
  public:

    PFBlockElementGsfTrack() {} 

    PFBlockElementGsfTrack(const GsfPFRecTrackRef& gsfref, 
			   const math::XYZTLorentzVector& Pin, 
			   const math::XYZTLorentzVector& Pout);

    PFBlockElement* clone() const { return new PFBlockElementGsfTrack(*this); }
    
    void Dump(std::ostream& out = std::cout, 
              const char* tab = " " ) const;

    /// \return tracktype
    virtual bool trackType(TrackType trType) const { 
      return (trackType_>>trType) & 1; 
    }
      
    /// \set the trackType
    virtual void setTrackType(TrackType trType, bool value) {
      if(value)  trackType_ = trackType_ | (1<<trType);
      else trackType_ = trackType_ ^ (1<<trType);
    }
    
    bool isSecondary() const { 
      return trackType(T_FROM_GAMMACONV); 
    }

    /// \return reference to the corresponding PFGsfRecTrack
    GsfPFRecTrackRef GsftrackRefPF() const {
      return GsftrackRefPF_;
    }
    
    /// \return reference to the corresponding GsfTrack
    reco::GsfTrackRef GsftrackRef() const {
      return GsftrackRef_;
    }
 
    
    
   
    /// \return position at ECAL entrance
    const math::XYZPointF& positionAtECALEntrance() const {
      return positionAtECALEntrance_;
    }
    


    const GsfPFRecTrack & GsftrackPF() const { return *GsftrackRefPF_;}
  
    const math::XYZTLorentzVector& Pin() const    { return Pin_; }
    const math::XYZTLorentzVector& Pout() const    { return Pout_; }
  private:
    


    /// reference to the corresponding GSF track (transient)
    GsfPFRecTrackRef  GsftrackRefPF_;
    
    /// reference to the corresponding GSF track 
    reco::GsfTrackRef GsftrackRef_;
    
    
    /// The CorrespondingKFTrackRef is needeed. 
    math::XYZTLorentzVector Pin_;
    math::XYZTLorentzVector Pout_;
    unsigned int  trackType_; 
 

   /// position at ECAL entrance
    math::XYZPointF        positionAtECALEntrance_;
    


  };
}

#endif

