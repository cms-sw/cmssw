#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <set>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerDTC {

  LayerEncoding::LayerEncoding(const ParameterSet& iConfig, const Setup* setup)
      : setup_(setup), numDTCsPerRegion_(setup->numDTCsPerRegion()) {
    encodingsLayerId_.reserve(numDTCsPerRegion_);
    for (int dtcInRegion = 0; dtcInRegion < setup->numDTCsPerRegion(); dtcInRegion++) {
      set<int> encodingLayerId;
      for (int region = 0; region < setup->numRegions(); region++) {
        const int dtcId = dtcInRegion + region * setup->numDTCsPerRegion();
        const vector<SensorModule*>& modules = setup->dtcModules(dtcId);
        for (SensorModule* sm : modules)
          encodingLayerId.insert(sm->layerId());
      }
      // check configuration
      if ((int)encodingLayerId.size() > setup->hybridNumLayers()) {
        cms::Exception exception("overflow");
        exception << "Cabling map connects more than " << setup->hybridNumLayers() << " layers to a DTC.";
        exception.addContext("trackerDTC::LayerEncoding::LayerEncoding");
        throw exception;
      }
      encodingsLayerId_.emplace_back(encodingLayerId.begin(), encodingLayerId.end());
    }
  }

  // decode layer id for given sensor module
  int LayerEncoding::decode(SensorModule* sm) const {
    const vector<int>& encoding = encodingsLayerId_.at(sm->dtcId() % setup_->numDTCsPerRegion());
    const auto pos = find(encoding.begin(), encoding.end(), sm->layerId());
    return distance(encoding.begin(), pos);
  }

}  // namespace trackerDTC
