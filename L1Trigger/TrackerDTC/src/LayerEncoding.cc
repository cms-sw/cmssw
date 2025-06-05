#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <set>
#include <algorithm>
#include <iterator>

namespace trackerDTC {

  LayerEncoding::LayerEncoding(const tt::Setup* setup) : setup_(setup), numDTCsPerRegion_(setup->numDTCsPerRegion()) {
    encodingsLayerId_.reserve(numDTCsPerRegion_);
    for (int dtcInRegion = 0; dtcInRegion < setup->numDTCsPerRegion(); dtcInRegion++) {
      std::set<int> encodingLayerId;
      for (int region = 0; region < setup->numRegions(); region++) {
        const int dtcId = dtcInRegion + region * setup->numDTCsPerRegion();
        const std::vector<tt::SensorModule*>& modules = setup->dtcModules(dtcId);
        for (tt::SensorModule* sm : modules)
          encodingLayerId.insert(sm->layerId());
      }
      // check configuration
      if (static_cast<int>(encodingLayerId.size()) > setup->hybridNumLayers()) {
        cms::Exception exception("overflow");
        exception << "Cabling map connects more than " << setup->hybridNumLayers() << " layers to a DTC.";
        exception.addContext("trackerDTC::LayerEncoding::LayerEncoding");
        throw exception;
      }
      encodingsLayerId_.emplace_back(encodingLayerId.begin(), encodingLayerId.end());
    }
  }

  // decode layer id for given sensor module
  int LayerEncoding::decode(const tt::SensorModule* sm) const {
    const std::vector<int>& encoding = encodingsLayerId_.at(sm->dtcId() % setup_->numDTCsPerRegion());
    const auto pos = std::find(encoding.begin(), encoding.end(), sm->layerId());
    return std::distance(encoding.begin(), pos);
  }

}  // namespace trackerDTC
