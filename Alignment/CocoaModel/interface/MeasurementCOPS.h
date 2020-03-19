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
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"

class MeasurementCOPS : public Measurement {
public:
  MeasurementCOPS(const ALIint measdim, ALIstring& type, ALIstring& name) : Measurement(measdim, type, name) {
    for (unsigned int ii = 0; ii < 4; ii++)
      theXlaserLine[ii] = -1;
  };
  MeasurementCOPS(){};
  ~MeasurementCOPS() override{};

  // Get simulated value (called every time a parameter is displaced)
  void calculateSimulatedValue(ALIbool firstTime) override;

  //---------- Add any correction between the measurement data and the default format in COCOA
  void correctValueAndSigma() override;

  //---------- Convert from V to rad
  void setConversionFactor(const std::vector<ALIstring>& wordlist) override;
  int xlaserLine(unsigned int ii) override { return theXlaserLine[ii]; }
  void setXlaserLine(unsigned int ii, int val) override { theXlaserLine[ii] = val; };

private:
  ALIdouble theDisplace[4];
  ALIint theXlaserLine
      [4];  // which x-hair laser line is measuring the CCD (0: X for UP and DOWN, Y for LEFT and RIGHT; 1: Y for UP and DOWN, X for LEFT and RIGHT ). Initialised at -1
};

#endif
