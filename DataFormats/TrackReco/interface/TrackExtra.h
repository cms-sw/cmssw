#ifndef TrackReco_TrackExtra_h
#define TrackReco_TrackExtra_h
/** \class reco::TrackExtra
 *
 * Extension of a reconstructed reco::Track. It is ment to be stored
 * in the RECO, and to be referenced by its corresponding
 * object stored in the AOD
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: TrackBase.h,v 1.2 2006/03/01 09:28:47 llista Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/RecHitFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

namespace reco {
  class TrackExtra {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// default constructor
    TrackExtra() { }
    /// constructor from outermost position and momentum
    TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok );
    /// outermost point
    const Point & outerPosition() const { return outerPosition_; }
    /// momentum vector at outermost point
    const Vector & outerMomentum() const { return outerMomentum_; }
    /// returns true if the outermost point is valid
    bool outerOk() const { return outerOk_; }
    /// add a reference to a RecHit
    void add( const RecHitRef & r ) { recHits_.push_back( r ); }
    /// first iterator over RecHits
    recHit_iterator recHitsBegin() const { return recHits_.begin(); }
    /// last iterator over RecHits
    recHit_iterator recHitsEnd() const { return recHits_.end(); }
    /// number of RecHits
    size_t recHitsSize() const { return recHits_.size(); }
    /// x coordinate of momentum vector at the outermost point    
    double outerPx() const { return outerMomentum_.X(); }
    /// y coordinate of momentum vector at the outermost point
    double outerPy() const { return outerMomentum_.Y(); }
    /// z coordinate of momentum vector at the outermost point
    double outerPz() const { return outerMomentum_.Z(); }
    /// x coordinate the outermost point
    double outerX() const { return outerPosition_.X(); }
    /// y coordinate the outermost point
    double outerY() const { return outerPosition_.Y(); }
    /// z coordinate the outermost point
    double outerZ() const { return outerPosition_.Z(); }
    /// magnitude of momentum vector at the outermost point    
    double outerP() const { return outerMomentum().R(); }
    /// transverse momentum at the outermost point    
    double outerPt() const { return outerMomentum().Rho(); }
    /// azimuthal angle of the  momentum vector at the outermost point
    double outerPhi() const { return outerMomentum().Phi(); }
    /// pseudorapidity the  momentum vector at the outermost point
    double outerEta() const { return outerMomentum().Eta(); }
    /// polar angle of the  momentum vector at the outermost point
    double outerTheta() const { return outerMomentum().Theta(); }
    /// polar radius of the outermost point
    double outerRadius() const { return outerPosition().Rho(); }

  private:
    /// outermost point
    Point outerPosition_;
    /// momentum vector at outermost point
    Vector outerMomentum_;
    /// outermost point validity flag
    bool outerOk_;
    /// references to RecHits
    RecHitRefs recHits_;
  };

}

#endif
