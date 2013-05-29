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

#define maxEta    14   // ieta rings for HE 
#define maxLay    19   // max.number of layers 
#define maxDepth  7    // with some safety margin (wrt 5)

class HERecalibration {

public:
  HERecalibration(double integrated_lumi);
  ~HERecalibration();

  double getCorr(int ieta, int idepth);
  void  setDsegm(std::vector<std::vector<int> > m_segmentation);

private:

  void initialize();
  double iLumi;
  HEDarkening darkening;

 // Tabulated mean energy values per layer and per depth
  double dsegm[maxEta][maxLay];
  double corr[maxEta][maxDepth];

};


#endif // HERecalibration_h
