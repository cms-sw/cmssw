#ifndef RecoTauTag_Pi0Tau_Tau3D_h
#define RecoTauTag_Pi0Tau_Tau3D_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0Fwd.h"

namespace reco {

  /**\class Tau3D
     \brief reconstructed Tau candidates using 3D angle instead of dR cone
     
     \author Dongwook Jang
     \date   December 2006
  */

  class Tau3D {

  public:

    Tau3D();

    Tau3D(const reco::TrackRef seedTrack,
	  const reco::TrackRefVector &trackColl,
	  const reco::Pi0Collection &pi0Coll);
  
    Tau3D(const Tau3D& other);

    // \return seed track
    reco::TrackRef seedTrack() const { return seedTrack_; }

    // \return collection of tracks in 30 degree cone without any threshold
    const reco::TrackRefVector &tracks() const { return tracks_; }

    // \return collection of pi0s in 30 degree cone without any threshold
    const reco::Pi0Collection &pi0s() const { return pi0s_; }

    // overwriting ostream <<
    friend  std::ostream& operator<<(std::ostream& out, 
				     const Tau3D& tau);

  private:
    
    // seed track ie. highest pt track
    reco::TrackRef seedTrack_;

    // collection of tracks in 30 degree cone without any threshold
    reco::TrackRefVector tracks_;

    // collection of pi0s in 30 degree cone without any threshold
    reco::Pi0Collection pi0s_;

  };

}
#endif
