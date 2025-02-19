// COCOA class header file
// Id:  MeasurementTiltmeter.h
// CAT: Model
//
// Class for measurements
// 
// History: v1.0 
// Authors:
//   Pedro Arce

#ifndef _MEASUREMENTTILTMETER_HH
#define _MEASUREMENTTILTMETER_HH

#include <vector>
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"

class MeasurementTiltmeter : public Measurement
{ 
public:
  MeasurementTiltmeter( const ALIint measdim, ALIstring& type, ALIstring& name ) : Measurement( measdim, type, name ), theFactor(1.), theFactorSigma(0.), theConstantTerm(0.), theConstantTermSigma(0.), thePedestal(0.), thePedestalSigma(0.)
{ };
  MeasurementTiltmeter(){ };   
  ~MeasurementTiltmeter(){ };
    
  // Get simulated value (called every time a parameter is displaced)
  virtual void calculateSimulatedValue( ALIbool firstTime );

  //---------- Convert from V to rad
  virtual void setConversionFactor( const std::vector<ALIstring>& wordlist );

  //---------- Add any correction between the measurement data and the default format in COCOA
  virtual void correctValueAndSigma();

 private:
  ALIdouble theFactor;
  ALIdouble theFactorSigma;
  ALIdouble theConstantTerm;
  ALIdouble theConstantTermSigma;
  ALIdouble thePedestal;
  ALIdouble thePedestalSigma;
};

#endif
