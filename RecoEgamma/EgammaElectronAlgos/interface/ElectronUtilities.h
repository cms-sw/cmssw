#ifndef RecoEgamma_EgammaElectronAlgos_ElectronUtilities_H
#define RecoEgamma_EgammaElectronAlgos_ElectronUtilities_H

#include <DataFormats/GeometryVector/interface/GlobalPoint.h>
#include <DataFormats/GeometryVector/interface/GlobalVector.h>
#include <DataFormats/Math/interface/Point3D.h>
#include <DataFormats/Math/interface/Vector3D.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>


//===============================================================
// For an stl collection of pointers, enforce the deletion
// of pointed objects in case of exception.
//===============================================================

template <typename StlColType>
class ExceptionSafeStlPtrCol : public StlColType
 {
  public :
    ExceptionSafeStlPtrCol() : StlColType() {}
    ~ExceptionSafeStlPtrCol()
     {
      typename StlColType::const_iterator it ;
      for ( it = StlColType::begin() ; it != StlColType::end() ; it++ )
       { delete (*it) ; }
     }
 } ;


//===============================================================
// Normalization of angles
//===============================================================

template <typename RealType>
RealType normalized_phi( RealType phi )
 {
  constexpr RealType pi(M_PI);
  constexpr RealType pi2(2*M_PI);
  if (phi>pi)  { phi -= pi2 ; }
  if (phi<-pi) { phi += pi2; }
  return phi ;
 }


//===============================================================
// Convert between flavors of points and vectors,
// assuming existence of x(), y() and z().
//===============================================================

template <typename Type1, typename Type2>
void ele_convert( const Type1 & obj1, Type2 & obj2 )
 { obj2 = Type2(obj1.x(),obj1.y(),obj1.z()) ; }


//===============================================================
// When wanting to compute and compare several characteristics of
// one or two points, relatively to a given origin
//===============================================================

class EleRelPoint
 {
  public :
    EleRelPoint( const math::XYZPoint & p, const math::XYZPoint & origin ) : relP_(p.x()-origin.x(),p.y()-origin.y(),p.z()-origin.z()) {}
    EleRelPoint( const GlobalPoint & p, const math::XYZPoint & origin ) : relP_(p.x()-origin.x(),p.y()-origin.y(),p.z()-origin.z()) {}
    EleRelPoint( const math::XYZPoint & p, const GlobalPoint & origin ) : relP_(p.x()-origin.x(),p.y()-origin.y(),p.z()-origin.z()) {}
    EleRelPoint( const GlobalPoint & p, const GlobalPoint & origin ) : relP_(p.x()-origin.x(),p.y()-origin.y(),p.z()-origin.z()) {}
    double eta() { return relP_.eta() ; }
    double phi() { return normalized_phi(relP_.phi()) ; }
    double perp() { return std::sqrt(relP_.x()*relP_.x()+relP_.y()*relP_.y()) ; }
  private :
    math::XYZVector relP_ ;
 } ;

class EleRelPointPair
 {
  public :
    EleRelPointPair( const math::XYZPoint & p1, const math::XYZPoint & p2, const math::XYZPoint & origin ) : relP1_(p1.x()-origin.x(),p1.y()-origin.y(),p1.z()-origin.z()), relP2_(p2.x()-origin.x(),p2.y()-origin.y(),p2.z()-origin.z()) {}
    EleRelPointPair( const GlobalPoint & p1, const math::XYZPoint & p2, const math::XYZPoint & origin ) : relP1_(p1.x()-origin.x(),p1.y()-origin.y(),p1.z()-origin.z()), relP2_(p2.x()-origin.x(),p2.y()-origin.y(),p2.z()-origin.z()) {}
    EleRelPointPair( const math::XYZPoint & p1, const GlobalPoint & p2, const math::XYZPoint & origin ) : relP1_(p1.x()-origin.x(),p1.y()-origin.y(),p1.z()-origin.z()), relP2_(p2.x()-origin.x(),p2.y()-origin.y(),p2.z()-origin.z()) {}
    EleRelPointPair( const math::XYZPoint & p1, const math::XYZPoint & p2, const GlobalPoint & origin ) : relP1_(p1.x()-origin.x(),p1.y()-origin.y(),p1.z()-origin.z()), relP2_(p2.x()-origin.x(),p2.y()-origin.y(),p2.z()-origin.z()) {}
    EleRelPointPair( const GlobalPoint & p1, const GlobalPoint & p2, const math::XYZPoint & origin ) : relP1_(p1.x()-origin.x(),p1.y()-origin.y(),p1.z()-origin.z()), relP2_(p2.x()-origin.x(),p2.y()-origin.y(),p2.z()-origin.z()) {}
    EleRelPointPair( const math::XYZPoint & p1, const GlobalPoint & p2, const GlobalPoint & origin ) : relP1_(p1.x()-origin.x(),p1.y()-origin.y(),p1.z()-origin.z()), relP2_(p2.x()-origin.x(),p2.y()-origin.y(),p2.z()-origin.z()) {}
    EleRelPointPair( const GlobalPoint & p1, const math::XYZPoint & p2, const GlobalPoint & origin ) : relP1_(p1.x()-origin.x(),p1.y()-origin.y(),p1.z()-origin.z()), relP2_(p2.x()-origin.x(),p2.y()-origin.y(),p2.z()-origin.z()) {}
    EleRelPointPair( const GlobalPoint & p1, const GlobalPoint & p2, const GlobalPoint & origin ) : relP1_(p1.x()-origin.x(),p1.y()-origin.y(),p1.z()-origin.z()), relP2_(p2.x()-origin.x(),p2.y()-origin.y(),p2.z()-origin.z()) {}
    auto dEta() { return (relP1_.eta()-relP2_.eta()) ; }
    auto dPhi() { return normalized_phi(relP1_.barePhi()-relP2_.barePhi()) ; }
    auto dZ() { return (relP1_.z()-relP2_.z()) ; }
    auto dPerp() { return (relP1_.perp()-relP2_.perp()) ; }
  private :
    GlobalVector relP1_ ;
    GlobalVector relP2_ ;
 } ;


//===============================================================
// Low level functions for the computing of characteristics
// relatively to a given origin. Not meant to improve
// performance, but rather for the easy later localization
// of all such transformations.
//===============================================================

template <typename PointType>
double relative_eta( const PointType & p, const PointType & origin )
 { return (p-origin).eta() ; }

template <typename PointType>
double relative_phi( const PointType & p, const PointType & origin )
 { return normalized_phi((p-origin).phi()) ; }


#endif

