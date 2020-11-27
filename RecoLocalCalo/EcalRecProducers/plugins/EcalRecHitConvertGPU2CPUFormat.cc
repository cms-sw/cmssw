#include <iostream>

#include "CUDADataFormats/EcalRecHitSoA/interface/EcalRecHit.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Common.h"

class EcalRecHitConvertGPU2CPUFormat : public edm::stream::EDProducer<> {
public:
  explicit EcalRecHitConvertGPU2CPUFormat(edm::ParameterSet const& ps);
  ~EcalRecHitConvertGPU2CPUFormat() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  using InputProduct = ecal::RecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  const edm::EDGetTokenT<InputProduct> recHitsGPUEB_;
  const edm::EDGetTokenT<InputProduct> recHitsGPUEE_;

  const std::string recHitsLabelCPUEB_, recHitsLabelCPUEE_;
};

void EcalRecHitConvertGPU2CPUFormat::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsLabelGPUEB", edm::InputTag("ecalRecHitProducerGPU", "EcalRecHitsGPUEB"));
  desc.add<edm::InputTag>("recHitsLabelGPUEE", edm::InputTag("ecalRecHitProducerGPU", "EcalRecHitsGPUEE"));

  desc.add<std::string>("recHitsLabelCPUEB", "EcalRecHitsEB");
  desc.add<std::string>("recHitsLabelCPUEE", "EcalRecHitsEE");

  confDesc.addWithDefaultLabel(desc);
}

EcalRecHitConvertGPU2CPUFormat::EcalRecHitConvertGPU2CPUFormat(const edm::ParameterSet& ps)
    : recHitsGPUEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsLabelGPUEB"))},
      recHitsGPUEE_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsLabelGPUEE"))},
      recHitsLabelCPUEB_{ps.getParameter<std::string>("recHitsLabelCPUEB")},
      recHitsLabelCPUEE_{ps.getParameter<std::string>("recHitsLabelCPUEE")} {
  produces<EBRecHitCollection>(recHitsLabelCPUEB_);
  produces<EERecHitCollection>(recHitsLabelCPUEE_);
}

EcalRecHitConvertGPU2CPUFormat::~EcalRecHitConvertGPU2CPUFormat() {}

void EcalRecHitConvertGPU2CPUFormat::produce(edm::Event& event, edm::EventSetup const& setup) {
  auto const& hRecHitsGPUEB = event.get(recHitsGPUEB_);
  auto const& hRecHitsGPUEE = event.get(recHitsGPUEE_);

  auto recHitsCPUEB = std::make_unique<EBRecHitCollection>();
  auto recHitsCPUEE = std::make_unique<EERecHitCollection>();
  recHitsCPUEB->reserve(hRecHitsGPUEB.energy.size());
  recHitsCPUEE->reserve(hRecHitsGPUEE.energy.size());

  //
  //     explicit EcalRecHit(const DetId& id, float energy, float time, uint32_t extra = 0, uint32_t flagBits = 0):
  //

  for (uint32_t i = 0; i < hRecHitsGPUEB.energy.size(); ++i) {
    //
    // Save only if energy is >= 0 !
    // This is extremely important because the channels that were supposed
    // to be excluded get "-1" as energy
    //

    if (hRecHitsGPUEB.energy[i] >= 0) {
      recHitsCPUEB->emplace_back(DetId{hRecHitsGPUEB.did[i]},
                                 hRecHitsGPUEB.energy[i],
                                 hRecHitsGPUEB.time[i],
                                 hRecHitsGPUEB.extra[i],
                                 hRecHitsGPUEB.flagBits[i]);
    }

    //       std::cout << " EB :: extra [" << i << "::" << hRecHitsGPUEB.energy.size() << "] = " << hRecHitsGPUEB.extra[i] << std::endl;

    //         (*recHitsCPUEB)[i].setJitterError(hRecHitsGPUEB.timeError[i]);
    //         auto const offset = i * EcalDataFrame::MAXSAMPLES;
    //         for (uint32_t sample=0; sample<EcalDataFrame::MAXSAMPLES; ++sample)
    //             (*recHitsCPUEB)[i].setOutOfTimeAmplitude(
    //                 sample, hRecHitsGPUEB.energysAll[offset + sample]);
  }

  for (uint32_t i = 0; i < hRecHitsGPUEE.energy.size(); ++i) {
    //
    // Save only if energy is >= 0 !
    // This is extremely important because the channels that were supposed
    // to be excluded get "-1" as energy
    //

    if (hRecHitsGPUEE.energy[i] >= 0) {
      recHitsCPUEE->emplace_back(DetId{hRecHitsGPUEE.did[i]},
                                 hRecHitsGPUEE.energy[i],
                                 hRecHitsGPUEE.time[i],
                                 hRecHitsGPUEE.extra[i],
                                 hRecHitsGPUEE.flagBits[i]);
    }

    //       std::cout << " EE :: extra [" << i << "::" << hRecHitsGPUEE.energy.size() << "] = " << hRecHitsGPUEE.extra[i] << std::endl;

    //         (*recHitsCPUEE)[i].setJitterError(hRecHitsGPUEE.timeError[i]);
    //         auto const offset = i * EcalDataFrame::MAXSAMPLES;
    //         for (uint32_t sample=0; sample<EcalDataFrame::MAXSAMPLES; ++sample)
    //             (*recHitsCPUEE)[i].setOutOfTimeAmplitude(
    //                 sample, hRecHitsGPUEE.energysAll[offset + sample]);
  }

  event.put(std::move(recHitsCPUEB), recHitsLabelCPUEB_);
  event.put(std::move(recHitsCPUEE), recHitsLabelCPUEE_);
}

DEFINE_FWK_MODULE(EcalRecHitConvertGPU2CPUFormat);
