#ifndef DETECTOR_DESCRIPTION_DD_DETECTOR_H
#define DETECTOR_DESCRIPTION_DD_DETECTOR_H

#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace dd4hep {
  class Detector;
}

namespace cms  {
  struct DDDetector {

    using Detector = dd4hep::Detector;

    Detector* description = nullptr;
    DDVectorsMap vectors;
    DDPartSelectionMap partsels;
    DDSpecParRegistry specpars;
  };
}

#endif
