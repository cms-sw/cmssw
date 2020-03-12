//   COCOA class header file
//Id:  DeviationsFromFileSensor2D.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
//
//   History: v1.0
//   Pedro Arce

#ifndef _DEVIATIONSTRAVERSINGSENSOR2D_HH
#define _DEVIATIONSTRAVERSINGSENSOR2D_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/DeviationSensor2D.h"

#include <vector>

class ALIFileIn;

typedef std::vector<std::vector<DeviationSensor2D*> > vvd;
typedef std::vector<DeviationSensor2D*> vd;

class DeviationsFromFileSensor2D {
public:
  //---------- Constructors / Destructor
  DeviationsFromFileSensor2D() {
    theOffsetX = 0.;
    theOffsetY = 0.;
  };
  ~DeviationsFromFileSensor2D(){};

  // read file
  void readFile(ALIFileIn& ifdevi);

  // get the deviation in the matrix corresponding to the intersection point (intersx, intersy)
  std::pair<ALIdouble, ALIdouble> getDevis(ALIdouble intersX, ALIdouble intersY);

  // set offsetX/Y
  void setOffset(ALIdouble offX, ALIdouble offY) {
    theOffsetX = offX;
    theOffsetY = offY;
  }

  // Access data
  static const ALIbool apply() { return theApply; }

  // Set data
  static void setApply(ALIbool val) { theApply = val; }

private:
  bool firstScanDir;
  int theScanSenseX, theScanSenseY;

  ALIuint theNPoints;
  vvd theDeviations;
  static ALIbool theApply;

  ALIint verbose;

  ALIdouble theOffsetX, theOffsetY;
};

#endif
