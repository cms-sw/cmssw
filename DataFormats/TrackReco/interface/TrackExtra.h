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
 * \version $Id: TrackExtra.h,v 1.15 2007/07/16 10:31:11 gpetrucc Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/TrackReco/interface/TrackExtraBase.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

namespace reco {
  class TrackExtra : public TrackExtraBase {
  public:
    /// parameter dimension
    enum { dimension = 5 };
    /// error matrix size
    enum { covarianceSize = dimension * ( dimension + 1 ) / 2 };
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
    TrackExtra( const Point & outerPosition, const Vector & outerMomentum, bool ok ,
		const Point & innerPosition, const Vector & innerMomentum, bool iok,
		const CovarianceMatrix& outerState, unsigned int outerId,
		const CovarianceMatrix& innerState, unsigned int innerId, 
		PropagationDirection& seedDir);
    /// outermost hit position
    const Point & outerPosition() const { return outerPosition_; }
    /// momentum vector at outermost hit position
    const Vector & outerMomentum() const { return outerMomentum_; }
    /// returns true if the outermost hit is valid
    bool outerOk() const { return outerOk_; }
    /// innermost hit position
    const Point & innerPosition() const { return innerPosition_; }
    /// momentum vector at innermost hit position
    const Vector & innerMomentum() const { return innerMomentum_; }
    /// returns true if the innermost hit is valid
    bool innerOk() const { return innerOk_; }
    /// x coordinate of momentum vector at the outermost hit position    
    double outerPx() const { return outerMomentum_.X(); }
    /// y coordinate of momentum vector at the outermost hit position
    double outerPy() const { return outerMomentum_.Y(); }
    /// z coordinate of momentum vector at the outermost hit position
    double outerPz() const { return outerMomentum_.Z(); }
    /// x coordinate the outermost hit position
    double outerX() const { return outerPosition_.X(); }
    /// y coordinate the outermost hit position
    double outerY() const { return outerPosition_.Y(); }
    /// z coordinate the outermost hit position
    double outerZ() const { return outerPosition_.Z(); }
    /// magnitude of momentum vector at the outermost hit position    
    double outerP() const { return outerMomentum().R(); }
    /// transverse momentum at the outermost hit position    
    double outerPt() const { return outerMomentum().Rho(); }
    /// azimuthal angle of the  momentum vector at the outermost hit position
    double outerPhi() const { return outerMomentum().Phi(); }
    /// pseudorapidity the  momentum vector at the outermost hit position
    double outerEta() const { return outerMomentum().Eta(); }
    /// polar angle of the  momentum vector at the outermost hit position
    double outerTheta() const { return outerMomentum().Theta(); }
    /// polar radius of the outermost hit position
    double outerRadius() const { return outerPosition().Rho(); }

    /// outermost trajectory state curvilinear errors
    CovarianceMatrix outerStateCovariance() const;
    /// innermost trajectory state curvilinear errors
    CovarianceMatrix innerStateCovariance() const;
    /// fill outermost trajectory state curvilinear errors
    CovarianceMatrix & fillOuter( CovarianceMatrix & v ) const;
    /// fill outermost trajectory state curvilinear errors
    CovarianceMatrix & fillInner( CovarianceMatrix & v ) const;
    /// DetId of the detector on which surface the outermost state is located
    unsigned int outerDetId() const { return outerDetId_; }
    /// DetId of the detector on which surface the innermost state is located
    unsigned int innerDetId() const { return innerDetId_; }
    // direction how the hits were sorted in the original seed
    PropagationDirection seedDirection() const {return seedDir_;}

  private:
    /// outermost hit position
    Point outerPosition_;
    /// momentum vector at outermost hit position
    Vector outerMomentum_;
    /// outermost hit validity flag
    bool outerOk_;
    /// outermost trajectory state curvilinear errors 
    Double32_t outerCovariance_[ covarianceSize ];
    unsigned int outerDetId_;


    /// innermost hit position
    Point innerPosition_;
    /// momentum vector at innermost hit position
    Vector innerMomentum_;
    /// innermost hit validity flag
    bool innerOk_;
    /// innermost trajectory state 
    Double32_t innerCovariance_[ covarianceSize ];
    unsigned int innerDetId_;

    PropagationDirection seedDir_;

  };

}

#endif
