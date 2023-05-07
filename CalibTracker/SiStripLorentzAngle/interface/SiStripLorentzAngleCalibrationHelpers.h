#ifndef CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleCalibrationHelper_h
#define CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleCalibrationHelper_h

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include <string>
#include "TString.h"

namespace siStripLACalibration {

  inline std::string moduleLocationType(const uint32_t& mod, const TrackerTopology* tTopo) {
    const SiStripDetId detid(mod);
    std::string subdet = "";
    unsigned int layer = 0;
    if (detid.subDetector() == SiStripDetId::TIB) {
      subdet = "TIB";
      layer = tTopo->layer(mod);
    }

    if (detid.subDetector() == SiStripDetId::TOB) {
      subdet = "TOB";
      layer = tTopo->layer(mod);
    }

    std::string type = (detid.stereo() ? "s" : "a");
    std::string d_l_t = Form("%s_L%d%s", subdet.c_str(), layer, type.c_str());

    if (layer == 0)
      return subdet;
    return d_l_t;
  }

}  // namespace siStripLACalibration
#endif
