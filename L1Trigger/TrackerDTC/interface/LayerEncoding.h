#ifndef L1Trigger_TrackerDTC_LayerEncoding_h
#define L1Trigger_TrackerDTC_LayerEncoding_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncodingRcd.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

namespace trackerDTC {

  /*! \class  trackerDTC::LayerEncoding
   *  \brief  Class to encode layer ids used between DTC and TFP in Hybrid
   *  \author Thomas Schuh
   *  \date   2021, April
   */
  class LayerEncoding {
  public:
    LayerEncoding() {}
    LayerEncoding(const edm::ParameterSet& iConfig, const tt::Setup* setup);
    ~LayerEncoding() {}
    // decode layer id for given sensor module
    int decode(tt::SensorModule* sm) const;
    // get encoded layers read by given DTC
    const std::vector<int>& layers(int dtcId) const { return encodingsLayerId_.at(dtcId % numDTCsPerRegion_); }

  private:
    // helper class to store configurations
    const tt::Setup* setup_;
    // No. of DTCs per detector phi nonant
    int numDTCsPerRegion_;
    // outer index = dtc id in region, inner index = encoded layerId, inner value = decoded layerId
    std::vector<std::vector<int>> encodingsLayerId_;
  };

}  // namespace trackerDTC

EVENTSETUP_DATA_DEFAULT_RECORD(trackerDTC::LayerEncoding, trackerDTC::LayerEncodingRcd);

#endif
