//   COCOA class header file
//Id:  LightRay.h
//CAT: Model
//
//   Utility class that starts reading the system description file
//                and contains the static data 
// 
//   History: v1.0 
//   Pedro Arce

#ifndef LIGHTRAY_H
#define LIGHTRAY_H

class OpticalObject;
class ALIPlane;

#include "CLHEP/Vector/ThreeVector.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h" 

class LightRay
{
public:
  //----- construct a default LigthRay
  LightRay();
// Make a light ray out of the centre_glob points of two OptO:'source' and 'pointLens'
  LightRay( OpticalObject* p1, OpticalObject* p2);
  LightRay( CLHEP::Hep3Vector& vec1, CLHEP::Hep3Vector& vec2 );
  ~LightRay() {};

//@@@@@@@@@@@@@@@@@@@@ Methods for each OptO
//----- Make a light ray out of the centre_glob and Z direction of one OptO: 'laser' or 'source'
  void startLightRay( OpticalObject* opto );

  //----- Intersect light ray with a plane and change thePoint to the intersection point
  void intersect( const ALIPlane& plane);

  //----- Intersect light ray with a OptO (intersect it with its plane perpendicular to Z) and change thePoint to the intersection point
  void intersect( const OpticalObject& opto );

  //-----  Intersect the LightRay with a plane and then change the direction from reflection on this plane
  void reflect( const ALIPlane& plane);

  //----- Deviate a LightRay because of refraction when it passes from a medium of refraction index  refra_ind1 to a medium of refraction index  refra_ind2
  void refract( const ALIPlane plate, const ALIdouble refra_ind1, const ALIdouble refra_ind2);

  //----- shift 
  void shiftAndDeviateWhileTraversing( const OpticalObject* opto, char behav );
   void shiftAndDeviateWhileTraversing( const OpticalObject* opto, ALIdouble shiftX, ALIdouble shiftY, ALIdouble shiftZ, ALIdouble deviX, ALIdouble deviY, ALIdouble deviZ );
   // void shiftAndDeviateWhileTraversing( ALIdouble shiftAxis1, ALIdouble shiftAxis2, ALIdouble deviAxis1, ALIdouble deviAxis2 );

// ACCESS DATA MEMBERS
  const CLHEP::Hep3Vector& point() const{
      return _point;
  } 
  const CLHEP::Hep3Vector& direction() const{
      return _direction;
  } 
  void dumpData(const ALIstring& str) const;

 // SET DATA MEMBERS
  void setDirection( const CLHEP::Hep3Vector& direc) {
       _direction = direc;
  } 
  void setPoint( const CLHEP::Hep3Vector& point) {
       _point = point;
  } 
  

private:
  //-------------- Methods common to several OptO
// Intersect a LightRay with the X-Y plane of the GlobalVectorFrame of an OptO
 public:
  CLHEP::Hep3Vector IntersectWithOptOPlane( const OpticalObject* optoplane);
  CLHEP::Hep3Vector IntersectWithPlane(const CLHEP::Hep3Vector& plane_point,
             const CLHEP::Hep3Vector& plane_normal);

 // private DATA MEMBERS
  CLHEP::Hep3Vector _direction;
  CLHEP::Hep3Vector _point;
};


#endif
