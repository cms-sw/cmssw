#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"

#include <memory>

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerLayerEncoding
   *  \brief  Class to produce KF layer encoding
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerLayerEncoding : public edm::ESProducer {
  public:
    ProducerLayerEncoding(const edm::ParameterSet& iConfig);
    ~ProducerLayerEncoding() override {}
    std::unique_ptr<LayerEncoding> produce(const DataFormatsRcd& rcd);

  private:
    edm::ESGetToken<DataFormats, DataFormatsRcd> esGetToken_;
  };

  ProducerLayerEncoding::ProducerLayerEncoding(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  std::unique_ptr<LayerEncoding> ProducerLayerEncoding::produce(const DataFormatsRcd& rcd) {
    const DataFormats* dataFormats = &rcd.get(esGetToken_);
    return std::make_unique<LayerEncoding>(dataFormats);
  }

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerLayerEncoding);
