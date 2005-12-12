// COCOA class header file
// Id:  MeasurementCOPS.h
// CAT: Model
//
// Class for measurements
// 
// History: v1.0 
// Authors:
//   Pedro Arce

#ifndef _MeasurementCOPS_h
#define _MeasurementCOPS_h

#include <vector>
#include "OpticalAlignment/CocoaModel/interface/Measurement.h"
#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"

class MeasurementCOPS : public Measurement
{ 
public:
  MeasurementCOPS( const ALIint measdim, std::vector<ALIstring>& wl ) : Measurement( measdim, wl ){
    for(uint ii=0; ii<4; ii++) theXlaserLine[ii] = -1; 
  };
  MeasurementCOPS(){ };   
  ~MeasurementCOPS(){ };
    
  // Get simulated value (called every time a parameter is displaced)
  virtual void calculateSimulatedValue( ALIbool firstTime );

  //---------- Add any correction between the measurement data and the default format in COCOA
  virtual void correctValueAndSigma();

  //---------- Convert from V to rad
  virtual void setConversionFactor( const std::vector<ALIstring>& wordlist );
  virtual int xlaserLine( uint ii) {
    return theXlaserLine[ii];
  }
  virtual void setXlaserLine( uint ii, int val) {
    theXlaserLine[ii] = val;};

 private:

  ALIdouble theDisplace[4];
  ALIint theXlaserLine[4]; // which x-hair laser line is measuring the CCD (0: X for UP and DOWN, Y for LEFT and RIGHT; 1: Y for UP and DOWN, X for LEFT and RIGHT ). Initialised at -1

};

#endif
