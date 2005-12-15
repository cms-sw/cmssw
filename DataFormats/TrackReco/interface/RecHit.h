#ifndef TrackReco_RecHit_h
#define TrackReco_RecHit_h
/*

Dummy class to be replaced by real RecHits

$Id: RecHit.h,v 1.2 2005/12/11 17:58:16 llista Exp $

*/
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/RecHitFwd.h"

namespace reco {

  class RecHit {
  public:
    typedef math::XYZPoint Point;
    RecHit() {}
    RecHit( const Point & p );
    const Point & position() const { return position_; }
    double localX() const { return position_.X(); }
    double localY() const { return position_.Y(); }
    double localZ() const { return position_.Z(); }
  private:
    Point position_;
  };

}

#endif
