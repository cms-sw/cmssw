#ifndef CalibCalorimetry_HcalAlgos_HFRecalibration_h
#define CalibCalorimetry_HcalAlgos_HFRecalibration_h
//
// Simple class with parameterized function provided by James Wetzel
// to compansate for darkening of HF QP fibers
// in Upgrade conditions
// correction = f (integrated lumi, depth, ieta)
//
#include <cmath>
#include <iostream>
#include <vector>

typedef std::vector<double> vecOfDoubles;

namespace edm {
  class ParameterSet;
}

class HFRecalibration {
public:
  HFRecalibration(const edm::ParameterSet& pset);
  ~HFRecalibration();
  double getCorr(int ieta, int idepth, double lumi);

  //Calibration factors only calculated for iEta between and including 30 and 41
  static const unsigned int loweriEtaBin = 30;
  static const unsigned int upperiEtaBin = 41;

private:
  //Container for holding parameters from cff file
  std::vector<double> HFParsAB[2][2];
  double reCalFactor = 1.0;
};

#endif  // HFRecalibration_h
