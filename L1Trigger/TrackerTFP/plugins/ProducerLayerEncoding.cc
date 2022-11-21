#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"

#include <memory>

using namespace std;
using namespace edm;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerLayerEncoding
   *  \brief  Class to produce KF layer encoding
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerLayerEncoding : public ESProducer {
  public:
    ProducerLayerEncoding(const ParameterSet& iConfig);
    ~ProducerLayerEncoding() override {}
    unique_ptr<LayerEncoding> produce(const LayerEncodingRcd& rcd);

  private:
    const ParameterSet iConfig_;
    ESGetToken<DataFormats, DataFormatsRcd> esGetToken_;
  };

  ProducerLayerEncoding::ProducerLayerEncoding(const ParameterSet& iConfig) : iConfig_(iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  unique_ptr<LayerEncoding> ProducerLayerEncoding::produce(const LayerEncodingRcd& rcd) {
    const DataFormats* dataFormats = &rcd.get(esGetToken_);
    return make_unique<LayerEncoding>(dataFormats);
  }

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerLayerEncoding);