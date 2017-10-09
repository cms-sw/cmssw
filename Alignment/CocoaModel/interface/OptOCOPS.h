//   COCOA class header file
//Id:  OptOCOPS.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OptOCOPS_hh
#define _OptOCOPS_hh

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;
#include "Alignment/CocoaModel/interface/ALILine.h"
class DeviationsFromFileSensor2D;

class OptOCOPS: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOCOPS(){ };
  OptOCOPS(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data), fdevi_from_file(0){ };
  ~OptOCOPS(){ };

  //---------- defaultBehaviour: make measurement 
  virtual void defaultBehaviour( LightRay& lightray, Measurement& meas );
  //---------- Make measurement 
  virtual void makeMeasurement( LightRay& lightray, Measurement& meas );
  //---------- Fast simulation of the light ray traversing
  virtual void fastTraversesLightRay( LightRay& lightray );

  // Get intersection in local coordinates
  ALIdouble* convertPointToLocalCoordinates( const CLHEP::Hep3Vector& point);

#ifdef COCOA_VIS
  virtual void fillVRML();
  virtual void fillIguana();
#endif
  void constructSolidShape();

 private:
  ALIdouble getMeasFromInters( ALILine& line_xhair, ALILine& ccd, CLHEP::Hep3Vector& cops_line );
  // Deviation values read from file
  DeviationsFromFileSensor2D* deviFromFile;
  ALIbool fdevi_from_file;
  ALILine ccds[4];
};

#endif
