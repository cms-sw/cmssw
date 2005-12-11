#ifndef TrackReco_TrackExtra_h
#define TrackReco_TrackExtra_h
//
// $Id: TrackExtra.h,v 1.1 2005/11/22 13:51:44 llista Exp $
//
// Definition of TrackExtra class for RECO
//
// Author: Luca Lista
//
#include <Rtypes.h>
#include "DataFormats/TrackReco/interface/Vector.h"
#include "DataFormats/TrackReco/interface/RecHitFwd.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefVector.h"
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Point3D.h>

namespace reco {
  class TrackExtra {
  public:
    TrackExtra() { }
    TrackExtra( double x, double y, double z,
		double px, double py, double pz,
		bool ok );
    HepGeom::Vector3D<double> outerPosition() const {
      return HepGeom::Vector3D<double>( outerX(), outerY(), outerY() );
    }
    HepGeom::Vector3D<double> outerMomentum() const {
      return HepGeom::Vector3D<double>( outerPx(), outerPy(), outerPz() );
    }
    bool outerOk() const { return outerOk_; }
    void add( const RecHitRef & r ) { recHits_.push_back( r ); }
    recHit_iterator recHitsBegin() const { return recHits_.begin(); }
    recHit_iterator recHitsEnd() const { return recHits_.end(); }
    size_t recHitsSize() const { return recHits_.size(); }
    double outerPx() const { return outerMomentum_.get<0>(); }
    double outerPy() const { return outerMomentum_.get<1>(); }
    double outerPz() const { return outerMomentum_.get<2>(); }
    double outerX() const { return outerPosition_.get<0>(); }
    double outerY() const { return outerPosition_.get<1>(); }
    double outerZ() const { return outerPosition_.get<2>(); }
    double outerP() const { return outerMomentum().mag(); }
    double outerPt() const { return outerMomentum().perp(); }
    double outerPhi() const { return outerMomentum().phi(); }
    double outerEta() const { return outerMomentum().eta(); }
    double outerTheta() const { return outerMomentum().theta(); }
    double outerRadius() const { return outerPosition().perp(); }

  private:
    typedef Vector3D Vector;
    typedef Vector3D Point;

    Point outerPosition_;
    Vector outerMomentum_;
    bool outerOk_;
    RecHitRefs recHits_;
  };

}

#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#endif
