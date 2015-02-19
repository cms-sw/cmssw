#include "CondFormats/EgammaObjects/src/headers.h"


namespace CondFormats_EgammaObjects {
  struct dictionary {
    ElectronLikelihoodCategoryData a;
 
    ElectronLikelihoodCalibration b;
    ElectronLikelihoodCalibration::Entry c;
    std::vector<ElectronLikelihoodCalibration::Entry> d;
    std::vector<ElectronLikelihoodCalibration::Entry>::iterator d1;
    std::vector<ElectronLikelihoodCalibration::Entry>::const_iterator d2;
    GBRTree e1;
    GBRForest e2;
    GBRTree2D e3;
    GBRForest2D e4;
    GBRTreeD e5;
    GBRForestD e6;    
    
  };
}

