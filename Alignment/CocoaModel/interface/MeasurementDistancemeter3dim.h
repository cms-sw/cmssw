// COCOA class header file
// Id:  MeasurementDistancemeter3dim.h
// CAT: Model
//
// Class for measurements
// 
// History: v1.0 
// Authors:
//   Pedro Arce

#ifndef _MeasurementDistancemeter3dim_HH
#define _MeasurementDistancemeter3dim_HH

#include <vector>
#include "OpticalAlignment/CocoaModel/interface/Measurement.h"
#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"

class MeasurementDistancemeter3dim : public Measurement
{ 
public:
  MeasurementDistancemeter3dim( const ALIint measdim, std::vector<ALIstring>& wl ) : Measurement( measdim, wl ), theFactor(1.), theFactorSigma(0.){ };
  MeasurementDistancemeter3dim(){ };   
  ~MeasurementDistancemeter3dim(){ };
    
  // Get simulated value (called every time a parameter is displaced)
  virtual void calculateSimulatedValue( ALIbool firstTime );

  //---------- Convert from V to rad
  virtual void setConversionFactor( const std::vector<ALIstring>& wordlist );

  //---------- Add any correction between the measurement data and the default format in COCOA
  virtual void correctValueAndSigma();

 private:
  ALIdouble theFactor;
  ALIdouble theFactorSigma;
};

#endif
