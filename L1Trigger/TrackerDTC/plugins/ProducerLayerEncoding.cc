#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"

#include <memory>

namespace trackerDTC {

  /*! \class  trackerDTC::ProducerLayerEncoding
   *  \brief  Class to produce layer encoding used between DTC and TFP in Hybrid
   *  \author Thomas Schuh
   *  \date   2021, April
   */
  class ProducerLayerEncoding : public edm::ESProducer {
  public:
    ProducerLayerEncoding(const edm::ParameterSet& iConfig);
    ~ProducerLayerEncoding() override {}
    std::unique_ptr<LayerEncoding> produce(const tt::SetupRcd& rcd);

  private:
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetToken_;
  };

  ProducerLayerEncoding::ProducerLayerEncoding(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  std::unique_ptr<LayerEncoding> ProducerLayerEncoding::produce(const tt::SetupRcd& rcd) {
    const tt::Setup* setup = &rcd.get(esGetToken_);
    return std::make_unique<LayerEncoding>(setup);
  }

}  // namespace trackerDTC

DEFINE_FWK_EVENTSETUP_MODULE(trackerDTC::ProducerLayerEncoding);
