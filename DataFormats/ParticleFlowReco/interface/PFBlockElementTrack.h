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

    PFBlockElementTrack(const PFRecTrackRef& ref);

    PFBlockElement* clone() const { return new PFBlockElementTrack(*this); }
    
    void Dump(std::ostream& out = std::cout, 
              const char* tab = " " ) const;

    /// \return tracktype
    virtual bool trackType(TrackType trType) const { return (trackType_>>trType) & 1; }

    /// \set the trackType
    virtual void setTrackType(TrackType trType, bool value) {
          if(value)  trackType_ = trackType_ | (1<<trType);
          else trackType_ = trackType_ ^ (1<<trType);
    }

    
    /// \return reference to the corresponding PFRecTrack
    PFRecTrackRef trackRefPF() const { return trackRefPF_; }
    
    /// \return reference to the corresponding Track
    reco::TrackRef trackRef() const { return trackRef_; }

    ///\ check if the track is secondary
    bool isSecondary() const { return trackType(T_FROM_NUCL) || trackType(T_FROM_GAMMACONV); }

    /// \return the nuclear interaction associated
    NuclearInteractionRef nuclearRef() const { return nuclInterRef_; }

    /// \set the ref to the nuclear interaction
    void setNuclearRef(const NuclearInteractionRef& niref, TrackType trType) { nuclInterRef_ = niref; setTrackType(trType,true); } 

   /// \return reference to the corresponding Muon
   reco::MuonRef muonRef() const { return muonRef_; }

   /// \set reference to the Muon
   void setMuonRef(const MuonRef& muref) { muonRef_=muref; setTrackType(MUON,true); }

   /// \return ref to original recoConversion
   ConversionRef convRef() const {return convRef_;} 

   /// \set the ref to  gamma conversion
   void setConversionRef(const ConversionRef& convRef, TrackType trType) { convRef_ = convRef; setTrackType(trType,true); } 



    
  private:

    /// reference to the corresponding track (transient)
    PFRecTrackRef  trackRefPF_;

    /// reference to the corresponding track 
    reco::TrackRef trackRef_;

    unsigned int  trackType_;

    /// reference to the corresponding pf nuclear interaction
    NuclearInteractionRef  nuclInterRef_;

    /// reference to the corresponding muon
    reco::MuonRef muonRef_;

    /// reference to reco conversion
    ConversionRef convRef_;    

  };
}

#endif

