#ifndef BeamSpot_BeamSpot_h
#define BeamSpot_BeamSpot_h
/** \class reco::BeamSpot
 *  
 * Reconstructed beam spot object which provides position, error, and
 * width of the beam position.
 *
 * \author Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
 *
 * \version $Id: BeamSpot.h,v 1.12 2011/04/06 20:31:35 venturia Exp $
 *
 */

#include <Rtypes.h>
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <string>
#include <sstream>


namespace reco {

  class BeamSpot {
  public:
	  
    /// beam spot flags
    enum BeamType { Unknown=-1, Fake=0, LHC=1, Tracker=2 };
    
    /// point in the space
    typedef math::XYZPoint Point;
    enum { dimension = 7 };
    typedef math::Error<dimension>::type CovarianceMatrix;
    enum { dim3 = 3 };
    typedef math::Error<dim3>::type Covariance3DMatrix;
    enum { resdim = 2 };
    typedef math::Error<resdim>::type ResCovMatrix;
	
    /// default constructor
    BeamSpot();

    /// constructor from values
    BeamSpot( const Point &point,
	      double sigmaZ,
	      double dxdz,
	      double dydz,
	      double BeamWidthX,
		  const CovarianceMatrix &error,
		  BeamType type = Unknown) { 
      position_ = point;
      sigmaZ_ = sigmaZ;
      dxdz_ = dxdz;
      dydz_ = dydz;
      BeamWidthX_ = BeamWidthX;
      BeamWidthY_ = BeamWidthX;
      error_ = error;
	  type_ = type;
	  emittanceX_ = emittanceY_ = 0;
	  betaStar_ = 0;
    };

	
    /// position 
    const Point & position() const { return position_; }
    /// x coordinate 
    double x0() const { return position_.X(); }
    /// y coordinate 
    double y0() const { return position_.Y(); }
    /// z coordinate 
    double z0() const { return position_.Z(); }

    /// x coordinate of the beeam spot position at a given z value (it takes into account the dxdz slope)
    double x(const double z) const { return x0() + dxdz() * (z - z0()); }
    /// y coordinate of the beeam spot position at a given z value (it takes into account the dydz slope)
    double y(const double z) const { return y0() + dydz() * (z - z0()); }
    /// position of the beam spot at a given z value (it takes into account the dxdz and dydz slopes)
    const Point position(const double z) const;
    //    const Point position(const double z) const {Point pos(x(z),y(z),z);    return pos;}


    /// sigma z
    double sigmaZ() const { return sigmaZ_; }
    /// dxdz slope 
    double dxdz() const { return dxdz_; }
    /// dydz slope 
    double dydz() const { return dydz_; }
    /// beam width X
    double BeamWidthX() const { return BeamWidthX_; }
	/// beam width Y
    double BeamWidthY() const { return BeamWidthY_; }
    /// error on x
    double x0Error() const { return sqrt( error_(0,0) ); }
    /// error on y
    double y0Error() const { return sqrt( error_(1,1) ); }
    /// error on z
    double z0Error() const { return sqrt( error_(2,2) ); }
    /// error on sigma z
    double sigmaZ0Error() const { return sqrt ( error_(3,3) ); }
    /// error on dxdz
    double dxdzError() const { return sqrt ( error_(4,4) ); }
    /// error on dydz
    double dydzError() const { return sqrt ( error_(5,5) ); }

    /// error on beam width X, assume error in X = Y
    double BeamWidthXError() const { return sqrt ( error_(6,6) );}
	/// error on beam width Y, assume error in X = Y
    double BeamWidthYError() const { return sqrt ( error_(6,6) );}
		
	///
	void setBeamWidthX( double v ) { BeamWidthX_ = v; }
	void setBeamWidthY( double v ) { BeamWidthY_ = v; }
	
    /// (i,j)-th element of error matrix
    double covariance( int i, int j) const {
      return error_(i,j);
    }
    /// return full covariance matrix of dim 7
    CovarianceMatrix covariance() const { return error_; }
    /// return only 3D position covariance matrix
    Covariance3DMatrix covariance3D() const {

		Covariance3DMatrix matrix;
		for (int j=0; j<3; j++) {
			for (int k=j; k<3; k++) {
				matrix(j,k) = error_(j,k);
			}
		}
      return matrix;
    };
	/// return beam type
	BeamType type() const { return type_; }
	/// set beam type
	void setType( BeamType type ) { type_ = type; }
	///
    Covariance3DMatrix rotatedCovariance3D() const;

	/// additional information
	double emittanceX() const { return emittanceX_; }
	double emittanceY() const { return emittanceY_; }
	double betaStar() const { return betaStar_; }
	double beamWidthFromBeta( double z, double e ) const {
		return sqrt( e*betaStar_*(1 + pow((z-position_.Z())/betaStar_,2) ) );
	}
	///
	void setEmittanceX( double v ) { emittanceX_ = v; }
	void setEmittanceY( double v ) { emittanceY_ = v; }
	void setbetaStar( double v ) { betaStar_ = v; }
	
    /// print information
    void print( std::stringstream& ss ) const;

  private:
	/// position
	Point position_;
	/// errors
	CovarianceMatrix error_;

	Double32_t sigmaZ_;
	Double32_t BeamWidthX_;
	Double32_t BeamWidthY_;
	Double32_t dxdz_;
	Double32_t dydz_;
	Double32_t emittanceX_;
	Double32_t emittanceY_;
	Double32_t betaStar_;
	
	BeamType type_;
	
  };
  ///
  std::ostream& operator<< ( std::ostream&, BeamSpot beam );

}

#endif
