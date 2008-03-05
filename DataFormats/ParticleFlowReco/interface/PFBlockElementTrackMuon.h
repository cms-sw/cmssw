#ifndef __PFBlockElementTrackMuon__
#define __PFBlockElementTrackMuon__

#include <iostream>

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

namespace reco {
  
  /// \brief Muon Element.
  /// 
  /// this class contains a reference to the reco::Muon
  /// and to a PFRecTrack 
  /// it is the daughter of the PFBlockElementTrack class
  /// (marcella bona 2008/02/20)

  class PFBlockElementTrackMuon : public PFBlockElementTrack {

  public:
    PFBlockElementTrackMuon() {} 
    
    PFBlockElementTrackMuon(const reco::MuonRef& muonref,
			    const PFRecTrackRef& ref);

    PFBlockElementTrack* clone() const { 
      return new PFBlockElementTrackMuon(*this); 
    }

    void Dump(std::ostream& out = std::cout, 
              const char* tab = " " ) const;
    
    /// \return reference to the corresponding Muon
    reco::MuonRef muonRef() const {
      return muonRef_;
    }

  private:

    /// reference to the corresponding muon 
    reco::MuonRef muonRef_;
  };
}

#endif

