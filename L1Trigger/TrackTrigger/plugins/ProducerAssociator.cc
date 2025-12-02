#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackTrigger/interface/Associator.h"

#include <memory>

namespace tt {

  /*! \class  tt::ProducerAssociator
   *  \brief  provides class to associate TrackingParticles with TTStubs and vice versa.
   *  \author Thomas Schuh
   *  \date   2025, Aug
   */
  class ProducerAssociator : public edm::ESProducer {
  public:
    ProducerAssociator(const edm::ParameterSet& iConfig);
    ~ProducerAssociator() override = default;
    std::unique_ptr<Associator> produce(const SetupRcd& rcd);

  private:
    Associator::Config config_;
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetToken_;
  };

  ProducerAssociator::ProducerAssociator(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
    config_.minLayers_ = iConfig.getParameter<int>("MinLayers");
    config_.minLayersPS_ = iConfig.getParameter<int>("MinLayersPS");
    config_.minLayersGood_ = iConfig.getParameter<int>("MinLayersGood");
    config_.minLayersGoodPS_ = iConfig.getParameter<int>("MinLayersGoodPS");
    config_.maxLayersBad_ = iConfig.getParameter<int>("MaxLayersBad");
    config_.maxLayersBadPS_ = iConfig.getParameter<int>("MaxLayersBadPS");
  }

  std::unique_ptr<Associator> ProducerAssociator::produce(const SetupRcd& rcd) {
    const tt::Setup* setup = &rcd.get(esGetToken_);
    return std::make_unique<Associator>(config_, setup);
  }

}  // namespace tt

DEFINE_FWK_EVENTSETUP_MODULE(tt::ProducerAssociator);
