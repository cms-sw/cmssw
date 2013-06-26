//   COCOA class header file
//Id:  OptOSensor2D.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOSENSOR2D_HH
#define _OPTOSENSOR2D_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;
class DeviationsFromFileSensor2D;

class OptOSensor2D: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOSensor2D(){ };
  OptOSensor2D(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data), fdevi_from_file(0){ };
  ~OptOSensor2D(){ };

  //---------- defaultBehaviour: make measurement 
  virtual void defaultBehaviour( LightRay& lightray, Measurement& meas );
  //---------- Make measurement 
  virtual void makeMeasurement( LightRay& lightray, Measurement& meas );
  //---------- Fast simulation of the light ray traversing
  virtual void fastTraversesLightRay( LightRay& lightray );
  //---------- Detailed simulation of the light ray traversing
  virtual void detailedTraversesLightRay( LightRay& lightray );

  // Create and fill an extra entry, checking if it has to be read from file
  virtual void fillExtraEntry( std::vector<ALIstring>& wordlist );

  // Get intersection in local coordinates
  ALIdouble* convertPointToLocalCoordinates( const CLHEP::Hep3Vector& point);

#ifdef COCOA_VIS
  virtual void fillVRML();
  virtual void fillIguana();
#endif
  void constructSolidShape();

 private:
  // Deviation values read from file
  DeviationsFromFileSensor2D* deviFromFile;
  ALIbool fdevi_from_file;
};

#endif
