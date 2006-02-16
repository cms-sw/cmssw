#ifndef TrackReco_TrackExtra_h
#define TrackReco_TrackExtra_h
//
// $Id: TrackExtra.h,v 1.3 2005/12/15 20:42:49 llista Exp $
//
// Definition of TrackExtra class for RECO
//
// Author: Luca Lista
//
#include <Rtypes.h>
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/RecHitFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

namespace reco {
  class TrackExtra {
  public:
    typedef math::XYZPoint Point;
    typedef math::XYZVector Vector;
    TrackExtra() { }
    TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok );
    const Point & outerPosition() const { return outerPosition_; }
    const Vector & outerMomentum() const { return outerMomentum_; }
    bool outerOk() const { return outerOk_; }
    void add( const RecHitRef & r ) { recHits_.push_back( r ); }
    recHit_iterator recHitsBegin() const { return recHits_.begin(); }
    recHit_iterator recHitsEnd() const { return recHits_.end(); }
    size_t recHitsSize() const { return recHits_.size(); }
    double outerPx() const { return outerMomentum_.X(); }
    double outerPy() const { return outerMomentum_.Y(); }
    double outerPz() const { return outerMomentum_.Z(); }
    double outerX() const { return outerPosition_.X(); }
    double outerY() const { return outerPosition_.Y(); }
    double outerZ() const { return outerPosition_.Z(); }
    double outerP() const { return outerMomentum().R(); }
    double outerPt() const { return outerMomentum().Rho(); }
    double outerPhi() const { return outerMomentum().Phi(); }
    double outerEta() const { return outerMomentum().Eta(); }
    double outerTheta() const { return outerMomentum().Theta(); }
    double outerRadius() const { return outerPosition().Rho(); }

  private:
    Point outerPosition_;
    Vector outerMomentum_;
    bool outerOk_;
    RecHitRefs recHits_;
  };

}

#endif
