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
#include "L1Trigger/TrackFindingTracklet/interface/HitPatternHelper.h"
#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"

#include <memory>

namespace hph {

  class ProducerHPH : public edm::ESProducer {
  public:
    ProducerHPH(const edm::ParameterSet& iConfig);
    ~ProducerHPH() override = default;
    std::unique_ptr<Setup> produce(const trackerDTC::SetupRcd& Rcd);

  private:
    Setup::Config iConfig_;
    edm::ESGetToken<trklet::Setup, trackerDTC::SetupRcd> esGetTokenSetup_;
    edm::ESGetToken<trklet::DataFormats, trackerDTC::SetupRcd> esGetTokenDataFormats_;
  };

  ProducerHPH::ProducerHPH(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetTokenSetup_ = cc.consumes();
    esGetTokenDataFormats_ = cc.consumes();
    const edm::ParameterSet& oldKFPSet = iConfig.getParameter<edm::ParameterSet>("oldKFPSet");
    iConfig_.hphDebug_ = iConfig.getParameter<bool>("hphDebug");
    iConfig_.useNewKF_ = iConfig.getParameter<bool>("useNewKF");
    iConfig_.chosenRofZ_ = oldKFPSet.getParameter<double>("ChosenRofZ");
    iConfig_.etaRegions_ = oldKFPSet.getParameter<std::vector<double>>("EtaRegions");
  }

  std::unique_ptr<Setup> ProducerHPH::produce(const trackerDTC::SetupRcd& Rcd) {
    const trklet::Setup& setupTT = Rcd.get(esGetTokenSetup_);
    const trklet::DataFormats& dataFormats = Rcd.get(esGetTokenDataFormats_);
    return std::make_unique<Setup>(iConfig_, setupTT, dataFormats);
  }

}  // namespace hph

DEFINE_FWK_EVENTSETUP_MODULE(hph::ProducerHPH);
