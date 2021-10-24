#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "L1Trigger/TrackerDTC/interface/LayerEncoding.h"

#include <memory>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerDTC {

  /*! \class  trackerDTC::ProducerLayerEncoding
   *  \brief  Class to produce layer encoding used between DTC and TFP in Hybrid
   *  \author Thomas Schuh
   *  \date   2021, April
   */
  class ProducerLayerEncoding : public ESProducer {
  public:
    ProducerLayerEncoding(const ParameterSet& iConfig);
    ~ProducerLayerEncoding() override {}
    unique_ptr<LayerEncoding> produce(const LayerEncodingRcd& rcd);

  private:
    const ParameterSet iConfig_;
    ESGetToken<Setup, SetupRcd> esGetToken_;
  };

  ProducerLayerEncoding::ProducerLayerEncoding(const ParameterSet& iConfig) : iConfig_(iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  unique_ptr<LayerEncoding> ProducerLayerEncoding::produce(const LayerEncodingRcd& rcd) {
    const Setup* setup = &rcd.get(esGetToken_);
    return make_unique<LayerEncoding>(iConfig_, setup);
  }

}  // namespace trackerDTC

DEFINE_FWK_EVENTSETUP_MODULE(trackerDTC::ProducerLayerEncoding);