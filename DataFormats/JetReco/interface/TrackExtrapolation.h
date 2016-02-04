#ifndef DataFormats_TrackReco_interface_TrackExtrapolation_h
#define DataFormats_TrackReco_interface_TrackExtrapolation_h
/* \class reco::TrackExtrapolation TrackExtrapolation.h DataFormats/TrackReco/interface/TrackExtrapolation.h
*
* This class represents the track state at several radii (specified by user in producer).
* It stores a TrackRef to the original track, as well as vectors of the positions, momenta,
* and directions of the track at the various radii. 
*
* \author Salvatore Rappoccio, JHU
*
*
*/

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

#include <vector>

namespace reco {
  class TrackExtrapolation {
  public:


    TrackExtrapolation() {}
    TrackExtrapolation( reco::TrackRef const & track,
			std::vector<bool> isValid,
			std::vector<TrackBase::Point> const & pos,
			std::vector<TrackBase::Vector> const & mom,
			std::vector<TrackBase::Vector> const & dir ) :
    track_(track)
    {
      isValid_.resize(isValid.size());
      copy(isValid.begin(), isValid.end(), isValid_.begin() );
      pos_.resize( pos.size() );
      copy( pos.begin(), pos.end(), pos_.begin() );
      mom_.resize( mom.size() );
      copy( mom.begin(), mom.end(), mom_.begin() );
      dir_.resize( dir.size() );
      copy( dir.begin(), dir.end(), dir_.begin() );
    }

    ~TrackExtrapolation() {}

    reco::TrackRef const &                 track()      const { return track_;}
    std::vector<bool> const &              isValid()    const { return isValid_;}
    std::vector<TrackBase::Point> const &  positions()  const { return pos_;}
    std::vector<TrackBase::Vector> const & momenta()    const { return mom_;}
    std::vector<TrackBase::Vector> const & directions() const { return dir_;}

  protected:
    reco::TrackRef    track_;
    std::vector<bool>                  isValid_;
    std::vector<TrackBase::Point>      pos_;
    std::vector<TrackBase::Vector>     mom_;
    std::vector<TrackBase::Vector>     dir_;
  };
}


#endif
