#include "CondFormats/EgammaObjects/src/headers.h"

namespace CondFormats_EgammaObjects {
  struct dictionary {
    ElectronLikelihoodCategoryData a;

    ElectronLikelihoodCalibration b;
    ElectronLikelihoodCalibration::Entry c;
    std::vector<ElectronLikelihoodCalibration::Entry> d;
    std::vector<ElectronLikelihoodCalibration::Entry>::iterator d1;
    std::vector<ElectronLikelihoodCalibration::Entry>::const_iterator d2;
  };
}  // namespace CondFormats_EgammaObjects
