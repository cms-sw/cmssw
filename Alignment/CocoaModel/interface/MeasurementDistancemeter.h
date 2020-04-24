// COCOA class header file
// Id:  MeasurementDistancemeter.h
// CAT: Model
//
// Class for measurements
// 
// History: v1.0 
// Authors:
//   Pedro Arce

#ifndef _MEASUREMENTDISTANCEMETER_HH
#define _MEASUREMENTDISTANCEMETER_HH

#include <vector>
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"

class MeasurementDistancemeter : public Measurement
{ 
public:
  MeasurementDistancemeter( const ALIint measdim, ALIstring& type, ALIstring& name )
    : Measurement( measdim, type, name ), theFactor(1.), theFactorSigma(0.){};
  MeasurementDistancemeter(){ };   
  ~MeasurementDistancemeter() override{ };
    
  // Get simulated value (called every time a parameter is displaced)
  void calculateSimulatedValue( ALIbool firstTime ) override;

  //---------- Convert from V to rad
  void setConversionFactor( const std::vector<ALIstring>& wordlist ) override;

  //---------- Add any correction between the measurement data and the default format in COCOA
  void correctValueAndSigma() override;

 private:
  ALIdouble theFactor;
  ALIdouble theFactorSigma;
};

#endif
