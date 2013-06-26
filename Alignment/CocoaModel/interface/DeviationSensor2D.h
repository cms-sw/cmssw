//   COCOA class header file
//Id:  DeviationSensor2D.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _DEVIATIONSensor2D_HH
#define _DEVIATIONSensor2D_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include <vector>

class DeviationSensor2D
{

public:
  //---------- Constructors / Destructor
  DeviationSensor2D(){ };
  DeviationSensor2D( ALIdouble posDimFactor, ALIdouble angDimFactor );
  ~DeviationSensor2D(){ };
 
  // read file
  void fillData( const std::vector<ALIstring>& wl );

  // Access data
  const ALIdouble& posX() {
    return thePosX;
  }
  const ALIdouble& posY() {
    return thePosY;
  }
  const ALIdouble& posErrX() {
    return thePosErrX;
  }
  const ALIdouble& posErrY() {
    return thePosErrY;
  }
  const ALIdouble& devX() {
    return theDevX;
  }
  const ALIdouble& devY() {
    return theDevY;
  }
  const ALIdouble& devErrX() {
    return theDevErrX;
  }
  const ALIdouble& devErrY() {
    return theDevErrY;
  }
  
 private:
  ALIdouble thePosX, thePosY;
  ALIdouble thePosErrX, thePosErrY;
  ALIdouble theDevX, theDevY;
  ALIdouble theDevErrX, theDevErrY;
  ALIdouble thePosDimFactor, theAngDimFactor;
};

#endif
