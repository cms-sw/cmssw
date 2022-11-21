//  This ESproducer produces configuration needed by HitPatternHelper
//
//  Created by J.Li on 1/23/21.
//

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "L1Trigger/TrackTrigger/interface/HitPatternHelper.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <memory>

using namespace std;
using namespace edm;

namespace hph {

  class ProducerHPH : public ESProducer {
  public:
    ProducerHPH(const ParameterSet& iConfig);
    ~ProducerHPH() override {}
    unique_ptr<Setup> produce(const SetupRcd& Rcd);

  private:
    const ParameterSet iConfig_;
    ESGetToken<tt::Setup, tt::SetupRcd> getTokenSetup_;
  };

  ProducerHPH::ProducerHPH(const ParameterSet& iConfig) : iConfig_(iConfig) {
    auto cc = setWhatProduced(this);
    getTokenSetup_ = cc.consumes();
  }

  unique_ptr<Setup> ProducerHPH::produce(const SetupRcd& Rcd) {
    const tt::Setup& setupTT = Rcd.get(getTokenSetup_);
    return make_unique<Setup>(iConfig_, setupTT);
  }

}  // namespace hph

DEFINE_FWK_EVENTSETUP_MODULE(hph::ProducerHPH);
