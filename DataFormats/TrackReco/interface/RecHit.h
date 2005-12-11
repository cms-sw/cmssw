#ifndef TrackReco_RecHit_h
#define TrackReco_RecHit_h
/*

Dummy class to be replaced by real RecHits

$Id: RecHit.h,v 1.1 2005/11/22 13:51:44 llista Exp $

*/
#include <Rtypes.h>

namespace reco {

  class RecHit {
  public:
    RecHit() {}
    RecHit( double x, double y, double z );
    double localX() const { return x_; }
    double localY() const { return y_; }
    double localZ() const { return z_; }
  private:
    Double32_t x_, y_, z_;
  };

}

#include "DataFormats/TrackReco/interface/RecHitFwd.h"

#endif
