#ifndef CalibCalorimetry_HERecalibration_h
#define CalibCalorimetry_HERecalibration_h
//
// Simple class with tabulated/parameterized function 
// to compansate for darkening attenuation of HE scintillators   
// in Upgrade conditions
// Evaluated on the basis of   SimG4CMS/Calo/ HEDarkening by K.Pedro (Maryland)
// correction = f (integrated lumi, depth, ieta)
//

#include <cmath>
#include <vector>
#include <iostream>
#include "DataFormats/HcalCalibObjects/interface/HEDarkening.h"

class HERecalibration {

public:
  HERecalibration(double integrated_lumi, double cutoff);
  ~HERecalibration();

  double getCorr(int ieta, int idepth);
  void  setDsegm(const std::vector<std::vector<int> >& m_segmentation);

private:
  // max number of HE relaibration depths
  static const unsigned int nDepths = 7; 
  
  void initialize();
  double iLumi;
  double cutoff_;
  HEDarkening darkening;

 // Tabulated mean energy values per layer and per depth
  double dsegm[HEDarkening::nEtaBins][HEDarkening::nScintLayers];
  double  corr[HEDarkening::nEtaBins][nDepths];

};


#endif // HERecalibration_h
