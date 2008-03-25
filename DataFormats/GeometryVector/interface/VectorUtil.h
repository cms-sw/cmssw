#ifndef GeometryVector_Geom_Util_h
#define GeometryVector_Geom_Util_h



#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Math/VectorUtil.h" 
#include <cmath>


namespace Geom {

  using ROOT::Math::VectorUtil::Phi_0_2pi;
  using ROOT::Math::VectorUtil::Phi_mpi_pi;
  
  inline double deltaPhi(double phi1, double phi2) { 
    double dphi = phi2-phi1; 
    if ( dphi > M_PI ) {
      dphi -= 2.0*M_PI;
    } else if ( dphi <= -M_PI ) {
      dphi += 2.0*M_PI;
    }
    return dphi;
  }
  
  
  /**
     Find aximutal Angle difference between two generic vectors ( v2.Phi() - v1.Phi() ) 
     The only requirements on the Vector classes is that they implement the Phi() method
     \param v1  Vector of any type implementing the Phi() operator
     \param v2  Vector of any type implementing the Phi() operator
     \return  Phi difference
     \f[ \Delta \phi = \phi_2 - \phi_1 \f]
  */
  template <class Vector1, class Vector2> 
  double deltaPhi( const Vector1 & v1, const Vector2 & v2) { 
    return deltaPhi(v1.phi(),v2.phi()); 
  }
  
  
  
  /**
     Find difference in pseudorapidity (Eta) and Phi betwen two generic vectors
     The only requirements on the Vector classes is that they implement the Phi() and Eta() method
     \param v1  Vector 1  
     \param v2  Vector 2
     \return   Angle between the two vectors
     \f[ \Delta R = \sqrt{  ( \Delta \phi )^2 + ( \Delta \eta )^2 } \f]
  */ 
  template <class Vector1, class Vector2> 
  double deltaR2( const Vector1 & v1, const Vector2 & v2) { 
    double dphi = deltaPhi(v1,v2); 
    double deta = v2.eta() - v1.eta(); 
    return dphi*dphi + deta*deta; 
  }
  template <class Vector1, class Vector2> 
  double deltaR( const Vector1 & v1, const Vector2 & v2) { 
    return std::sqrt( deltaR2(v1,v2));
  }
  
}

#endif
