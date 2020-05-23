// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// algorithm specific
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit_soa.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/Common.h"

#include <iostream>

class EcalUncalibRecHitConvertGPU2CPUFormat : public edm::stream::EDProducer<> {
public:
  explicit EcalUncalibRecHitConvertGPU2CPUFormat(edm::ParameterSet const& ps);
  ~EcalUncalibRecHitConvertGPU2CPUFormat() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  using GPURecHitType = ecal::UncalibratedRecHit<ecal::Tag::soa>;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  const edm::EDGetTokenT<ecal::SoAUncalibratedRecHitCollection> recHitsGPUEB_;
  const edm::EDGetTokenT<ecal::SoAUncalibratedRecHitCollection> recHitsGPUEE_;

  const std::string recHitsLabelCPUEB_, recHitsLabelCPUEE_;
};

void EcalUncalibRecHitConvertGPU2CPUFormat::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsLabelGPUEB", edm::InputTag("ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEB"));
  desc.add<edm::InputTag>("recHitsLabelGPUEE", edm::InputTag("ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEE"));

  desc.add<std::string>("recHitsLabelCPUEB", "EcalUncalibRecHitsEB");
  desc.add<std::string>("recHitsLabelCPUEE", "EcalUncalibRecHitsEE");

  std::string label = "ecalUncalibRecHitConvertGPU2CPUFormat";
  confDesc.add(label, desc);
}

EcalUncalibRecHitConvertGPU2CPUFormat::EcalUncalibRecHitConvertGPU2CPUFormat(const edm::ParameterSet& ps)
    : recHitsGPUEB_{consumes<ecal::SoAUncalibratedRecHitCollection>(
          ps.getParameter<edm::InputTag>("recHitsLabelGPUEB"))},
      recHitsGPUEE_{
          consumes<ecal::SoAUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("recHitsLabelGPUEE"))},
      recHitsLabelCPUEB_{ps.getParameter<std::string>("recHitsLabelCPUEB")},
      recHitsLabelCPUEE_{ps.getParameter<std::string>("recHitsLabelCPUEE")} {
  produces<EBUncalibratedRecHitCollection>(recHitsLabelCPUEB_);
  produces<EEUncalibratedRecHitCollection>(recHitsLabelCPUEE_);
}

EcalUncalibRecHitConvertGPU2CPUFormat::~EcalUncalibRecHitConvertGPU2CPUFormat() {}

void EcalUncalibRecHitConvertGPU2CPUFormat::produce(edm::Event& event, edm::EventSetup const& setup) {
  edm::Handle<ecal::SoAUncalibratedRecHitCollection> hRecHitsGPUEB, hRecHitsGPUEE;
  event.getByToken(recHitsGPUEB_, hRecHitsGPUEB);
  event.getByToken(recHitsGPUEE_, hRecHitsGPUEE);

  auto recHitsCPUEB = std::make_unique<EBUncalibratedRecHitCollection>();
  auto recHitsCPUEE = std::make_unique<EEUncalibratedRecHitCollection>();
  recHitsCPUEB->reserve(hRecHitsGPUEB->amplitude.size());
  recHitsCPUEE->reserve(hRecHitsGPUEE->amplitude.size());

  for (uint32_t i = 0; i < hRecHitsGPUEB->amplitude.size(); ++i) {
    recHitsCPUEB->emplace_back(DetId{hRecHitsGPUEB->did[i]},
                               hRecHitsGPUEB->amplitude[i],
                               hRecHitsGPUEB->pedestal[i],
                               hRecHitsGPUEB->jitter[i],
                               hRecHitsGPUEB->chi2[i],
                               hRecHitsGPUEB->flags[i]);
    (*recHitsCPUEB)[i].setJitterError(hRecHitsGPUEB->jitterError[i]);
    auto const offset = i * EcalDataFrame::MAXSAMPLES;
    for (uint32_t sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample)
      (*recHitsCPUEB)[i].setOutOfTimeAmplitude(sample, hRecHitsGPUEB->amplitudesAll[offset + sample]);
  }

  for (uint32_t i = 0; i < hRecHitsGPUEE->amplitude.size(); ++i) {
    recHitsCPUEE->emplace_back(DetId{hRecHitsGPUEE->did[i]},
                               hRecHitsGPUEE->amplitude[i],
                               hRecHitsGPUEE->pedestal[i],
                               hRecHitsGPUEE->jitter[i],
                               hRecHitsGPUEE->chi2[i],
                               hRecHitsGPUEE->flags[i]);
    (*recHitsCPUEE)[i].setJitterError(hRecHitsGPUEE->jitterError[i]);
    auto const offset = i * EcalDataFrame::MAXSAMPLES;
    for (uint32_t sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample)
      (*recHitsCPUEE)[i].setOutOfTimeAmplitude(sample, hRecHitsGPUEE->amplitudesAll[offset + sample]);
  }

  event.put(std::move(recHitsCPUEB), recHitsLabelCPUEB_);
  event.put(std::move(recHitsCPUEE), recHitsLabelCPUEE_);
}

DEFINE_FWK_MODULE(EcalUncalibRecHitConvertGPU2CPUFormat);
