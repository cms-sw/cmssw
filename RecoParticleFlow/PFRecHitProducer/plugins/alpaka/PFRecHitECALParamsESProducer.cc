#include <memory>

#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsRecord.h"
#include "CalorimeterDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace particleFlowRecHitProducer;

  class PFRecHitECALParamsESProducer : public ESProducer {
  public:
    PFRecHitECALParamsESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig), cleaningThreshold_(iConfig.getParameter<double>("cleaningThreshold")) {
      auto cc = setWhatProduced(this);
      thresholdsToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<double>("cleaningThreshold", 2);
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<reco::PFRecHitECALParamsHostCollection> produce(const EcalPFRecHitThresholdsRcd& iRecord) {
      const auto& thresholds = iRecord.get(thresholdsToken_);
      auto product = std::make_unique<reco::PFRecHitECALParamsHostCollection>(ECAL::kSize, cms::alpakatools::host());
      for (uint32_t denseId = 0; denseId < ECAL::Barrel::kSize; denseId++)
        product->view().energyThresholds()[denseId] = thresholds.barrel(denseId);
      for (uint32_t denseId = 0; denseId < ECAL::Endcap::kSize; denseId++)
        product->view().energyThresholds()[denseId + ECAL::Barrel::kSize] = thresholds.endcap(denseId);
      product->view().cleaningThreshold() = cleaningThreshold_;
      return product;
    }

  private:
    const double cleaningThreshold_;
    edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> thresholdsToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitECALParamsESProducer);
