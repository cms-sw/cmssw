#ifndef L1Trigger_TrackerTFP_LayerEncodingRcd_h
#define L1Trigger_TrackerTFP_LayerEncodingRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "L1Trigger/TrackerTFP/interface/DataFormatsRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace trackerTFP {

  typedef edm::mpl::Vector<DataFormatsRcd> RcdsLayerEncoding;

  // record of trackerTFP::LayerEncoding
  class LayerEncodingRcd : public edm::eventsetup::DependentRecordImplementation<LayerEncodingRcd, RcdsLayerEncoding> {
  };

}  // namespace trackerTFP

#endif