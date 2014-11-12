#ifndef CalibCalorimetry_HBRecalibration_h
#define CalibCalorimetry_HBRecalibration_h
//
// Simple class with tabulated/parameterized function 
// to compansate for darkening attenuation of HE scintillators   
// in Upgrade conditions
// Evaluated on the basis of DataFormats/HcalCalibObjects/HBDarkening by K.Pedro (Maryland)
// correction = f (integrated lumi, depth, ieta)
//

#include <cmath>
#include <vector>
#include <iostream>
#include "DataFormats/HcalCalibObjects/interface/HBDarkening.h"

class HBRecalibration {

public:
  HBRecalibration(double integrated_lumi, double cutoff, unsigned int scenario);
  ~HBRecalibration();

  double getCorr(int ieta, int idepth);
  void  setDsegm(const std::vector<std::vector<int> >& m_segmentation);

private:
  // max number of HB recalibration depths
  static const unsigned int nDepths = 7; 
  
  void initialize();
  double iLumi;
  double cutoff_;
  HBDarkening darkening;

 // Tabulated mean energy values per layer and per depth
  double dsegm[HBDarkening::nEtaBins][HBDarkening::nScintLayers];
  double  corr[HBDarkening::nEtaBins][nDepths];

};


#endif // HBRecalibration_h
