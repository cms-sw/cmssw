// COCOA class header file
// Id:  MeasurementSensor2D.h
// CAT: Model
//
// Class for measurements
// 
// History: v1.0 
// Authors:
//   Pedro Arce

#ifndef _MEASUREMENTSENSOR2D_HH
#define _MEASUREMENTSENSOR2D_HH

#include <vector>
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"

class MeasurementSensor2D : public Measurement
{ 
public:
  MeasurementSensor2D( const ALIint measdim, ALIstring& type, ALIstring& name ) : Measurement( measdim, type, name ){ };
  MeasurementSensor2D(){ };   
  ~MeasurementSensor2D(){ };
    
  // Get simulated value (called every time a parameter is displaced)
  virtual void calculateSimulatedValue( ALIbool firstTime );

  //---------- Add any correction between the measurement data and the default format in COCOA
  virtual void correctValueAndSigma();

  //---------- Convert from V to rad
  virtual void setConversionFactor( const std::vector<ALIstring>& wordlist );

 private:

  ALIdouble theDisplaceX, theDisplaceY;
  ALIdouble theMultiplyX, theMultiplyY;
};

#endif
