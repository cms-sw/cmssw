#ifndef TrackReco_HelixParameters_h
#define TrackReco_HelixParameters_h
/*----------------------------
 $Id: HelixParameters.h,v 1.3 2005/11/24 12:12:08 llista Exp $
 Helix Track Parametrization

 Author: Luca Lista, INFN
------------------------------*/
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Vector.h"
#include <cmath>

namespace reco {
  namespace helix {
    enum index { i_d0 = 0, i_phi0, i_omega, i_dz, i_tanDip }; 
    typedef math::Error<6> PosMomError;
    typedef math::XYZVector Vector;
    typedef math::XYZPoint Point;
    typedef math::Vector<5> ParameterVector;
    typedef math::Error<5> ParameterError;
 
    class Parameters {
    public:
      Parameters() { }
      Parameters( const double * par ) : par_( par ) { }
      typedef ParameterVector::index index;
      /* removed for consistency with Covariance. See comment below...
      template<index i>
      double get() const { return par_.get<i>(); }
      template<index i>
      double & get() { return par_.get<i>(); }
      */
      double operator()( index i ) const { return par_( i ); }
      double & operator()( index i ) { return par_( i ); }
      double d0() const { return par_.get<i_d0>(); }
      double phi0() const { return par_.get<i_phi0>(); }
      double omega() const { return par_.get<i_omega>(); }
      double dz() const { return par_.get<i_dz>(); }
      double tanDip() const { return par_.get<i_tanDip>(); }
      double & d0() { return par_.get<i_d0>(); }
      double & phi0() { return par_.get<i_phi0>(); }
      double & omega() { return par_.get<i_omega>(); }
      double & dz() { return par_.get<i_dz>(); }
      double & tanDip() { return par_.get<i_tanDip>(); }
      int charge() const;
      double pt() const;
      Vector momentum() const;
      Point vertex() const;
      
    private:
      ParameterVector par_;
    };
  
    class Covariance {
    public:
      Covariance() {} 
      Covariance( const double * cov ) : cov_( cov ) { }
      typedef ParameterError::index index;
      /* those methods templates don't compile under LCG reflex dicts.
      template<index i, index j>
      double get() const { return cov_.get<i, j>(); }
      template<index i, index j>
      double & get() { return cov_.get<i, j>(); }
      */
      double operator()( index i, index j ) const { return cov_( i, j ); }
      double & operator()( index i, index j ) { return cov_ ( i, j ); }
      double d0Error() const { return sqrt( cov_.get<i_d0, i_d0>() ); }
      double phi0Error() const { return sqrt( cov_.get<i_phi0, i_phi0>() ); }
      double omegaError() const { return sqrt( cov_.get<i_omega, i_omega>() ); }
      double dzError() const { return sqrt( cov_.get<i_dz, i_dz>() ); }
      double tanDipError() const { return sqrt( cov_.get<i_tanDip, i_tanDip>() ); }

    private:
      ParameterError cov_;
    };
    
    void setFromCartesian( int q, const Point &, const Vector &, 
			   const PosMomError & ,
			   Parameters &, Covariance & ); 

    PosMomError posMomError( const Parameters &, const Covariance & );

    inline int Parameters::charge() const {
      return omega() > 0 ? +1 : -1;
    }
    
    inline double Parameters::pt() const {
      return 1./ fabs( omega() );
    }
    
    inline Vector Parameters::momentum() const {
      double p_t = pt();
      return Vector( - p_t * sin( phi0() ), p_t * cos( phi0() ), p_t * tanDip() );
    }
    
    inline Point Parameters::vertex() const {
      return Point( d0() * cos( phi0() ), d0() * sin( phi0() ), dz() );
    }

  }
}


#endif
