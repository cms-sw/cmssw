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

#define nDepths  7    // with some safety margin (wrt 5)

class HERecalibration {

public:
  HERecalibration(double integrated_lumi, double cutoff);
  ~HERecalibration();

  double getCorr(int ieta, int idepth);
  void  setDsegm(const std::vector<std::vector<int> >& m_segmentation);

private:

  void initialize();
  double iLumi;
  double cutoff_;
  HEDarkening darkening;

 // Tabulated mean energy values per layer and per depth
  double dsegm[nEtaBins][nScintLayers];
  double corr[nEtaBins][nDepths];

};


#endif // HERecalibration_h
