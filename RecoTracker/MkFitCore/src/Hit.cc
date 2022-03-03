#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "Matrix.h"

namespace mkfit {

  void MCHitInfo::reset() {}

  void print(std::string_view label, const MeasurementState& s) {
    std::cout << label << std::endl;
    std::cout << "x: " << s.parameters()[0] << " y: " << s.parameters()[1] << " z: " << s.parameters()[2] << std::endl
              << "errors: " << std::endl;
    dumpMatrix(s.errors());
    std::cout << std::endl;
  }

}  // end namespace mkfit
