#ifndef TrackReco_TrackExtra_h
#define TrackReco_TrackExtra_h
/** \class reco::TrackExtra TrackExtra.h DataFormats/TrackReco/interface/TrackExtra.h
 *
 * Extension of a reconstructed Track. It is ment to be stored
 * in the RECO, and to be referenced by its corresponding
 * object stored in the AOD
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: TrackExtra.h,v 1.9 2006/07/18 16:17:32 namapane Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/TrackReco/interface/TrackExtraBase.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

namespace reco {
  class TrackExtra : public TrackExtraBase {
  public:
    /// point in the space
    typedef math::XYZPoint Point;
    /// spatial vector
    typedef math::XYZVector Vector;
    /// 5 parameter covariance matrix
    typedef math::Error<5>::type CovarianceMatrix;
    /// index type
    typedef unsigned int index;

    /// default constructor
    TrackExtra() { }
    /// constructor from outermost position and momentum
    TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok,
		const CovarianceMatrix& outerState, unsigned int outerId);
    TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok ,
		const Point & innerPosition, const Vector & innerMomentum, bool iok,
		const CovarianceMatrix& outerState, unsigned int outerId,
		const CovarianceMatrix& innerState, unsigned int innerId);
    /// outermost point
    const Point & outerPosition() const { return outerPosition_; }
    /// momentum vector at outermost point
    const Vector & outerMomentum() const { return outerMomentum_; }
    /// returns true if the outermost point is valid
    bool outerOk() const { return outerOk_; }
    /// innermost point
    const Point & innerPosition() const { return innerPosition_; }
    /// momentum vector at innermost point
    const Vector & innerMomentum() const { return innerMomentum_; }
    /// returns true if the innermost point is valid
    bool innerOk() const { return innerOk_; }
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

    /// outermost trajectory state curvilinear errors
    CovarianceMatrix outerStateCovariance() const { return covariance( outerCovariance_);}
    /// innermost trajectory state curvilinear errors
    CovarianceMatrix innerStateCovariance() const { return covariance( innerCovariance_);}

    /// DetId of the detector on which surface the outermost state is located
    unsigned int outerDetId() const {return  outerDetId_;}
    /// DetId of the detector on which surface the innermost state is located
    unsigned int innerDetId() const {return  innerDetId_;}

  private:
    /// outermost point
    Point outerPosition_;
    /// momentum vector at outermost point
    Vector outerMomentum_;
    /// outermost point validity flag
    bool outerOk_;
    /// outermost trajectory state curvilinear errors 
    Double32_t outerCovariance_[15];
    unsigned int outerDetId_;


    /// innermost point
    Point innerPosition_;
    /// momentum vector at innermost point
    Vector innerMomentum_;
    /// innermost point validity flag
    bool innerOk_;
    /// innermost trajectory state 
    Double32_t innerCovariance_[15];
    unsigned int innerDetId_;
    

    CovarianceMatrix covariance( const Double32_t * data) const;

  };

}

#endif
