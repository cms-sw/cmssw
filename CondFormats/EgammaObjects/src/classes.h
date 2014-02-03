#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCategoryData.h"
#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCalibration.h"
#include "CondFormats/EgammaObjects/interface/GBRTree.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "CondFormats/EgammaObjects/interface/GBRTree2D.h"
#include "CondFormats/EgammaObjects/interface/GBRForest2D.h"

namespace {
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
    
  };
}
