#include "CUDADataFormats/EcalRecHitSoA/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EcalUncalibRecHitConvertGPU2CPUFormat : public edm::stream::EDProducer<> {
public:
  explicit EcalUncalibRecHitConvertGPU2CPUFormat(edm::ParameterSet const& ps);
  ~EcalUncalibRecHitConvertGPU2CPUFormat() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  using InputProduct = ecal::UncalibratedRecHit<calo::common::VecStoragePolicy<calo::common::CUDAHostAllocatorAlias>>;
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  bool produceEE_;
  const edm::EDGetTokenT<InputProduct> recHitsGPUEB_;
  edm::EDGetTokenT<InputProduct> recHitsGPUEE_;

  const std::string recHitsLabelCPUEB_;
  std::string recHitsLabelCPUEE_;
};

void EcalUncalibRecHitConvertGPU2CPUFormat::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsLabelGPUEB", edm::InputTag("ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEB"));

  desc.add<std::string>("recHitsLabelCPUEB", "EcalUncalibRecHitsEB");

  desc.add<bool>("produceEE", true);

  desc.add<edm::InputTag>("recHitsLabelGPUEE", edm::InputTag("ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEE"));
  desc.add<std::string>("recHitsLabelCPUEE", "EcalUncalibRecHitsEE");

  confDesc.add("ecalUncalibRecHitConvertGPU2CPUFormat", desc);
}

EcalUncalibRecHitConvertGPU2CPUFormat::EcalUncalibRecHitConvertGPU2CPUFormat(const edm::ParameterSet& ps)
    : produceEE_{ps.getParameter<bool>("produceEE")},
      recHitsGPUEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsLabelGPUEB"))},
      recHitsLabelCPUEB_{ps.getParameter<std::string>("recHitsLabelCPUEB")} {
  produces<EBUncalibratedRecHitCollection>(recHitsLabelCPUEB_);
  if (produceEE_) {
    recHitsGPUEE_ = consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsLabelGPUEE"));
    recHitsLabelCPUEE_ = ps.getParameter<std::string>("recHitsLabelCPUEE");
    produces<EEUncalibratedRecHitCollection>(recHitsLabelCPUEE_);
  }
}

EcalUncalibRecHitConvertGPU2CPUFormat::~EcalUncalibRecHitConvertGPU2CPUFormat() {}

void EcalUncalibRecHitConvertGPU2CPUFormat::produce(edm::Event& event, edm::EventSetup const& setup) {
  auto const& recHitsGPUEB = event.get(recHitsGPUEB_);
  auto recHitsCPUEB = std::make_unique<EBUncalibratedRecHitCollection>();
  recHitsCPUEB->reserve(recHitsGPUEB.amplitude.size());

  for (uint32_t i = 0; i < recHitsGPUEB.amplitude.size(); ++i) {
    recHitsCPUEB->emplace_back(DetId{recHitsGPUEB.did[i]},
                               recHitsGPUEB.amplitude[i],
                               recHitsGPUEB.pedestal[i],
                               recHitsGPUEB.jitter[i],
                               recHitsGPUEB.chi2[i],
                               recHitsGPUEB.flags[i]);
    (*recHitsCPUEB)[i].setAmplitudeError(recHitsGPUEB.amplitudeError[i]);
    (*recHitsCPUEB)[i].setJitterError(recHitsGPUEB.jitterError[i]);
    auto const offset = i * EcalDataFrame::MAXSAMPLES;
    for (uint32_t sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample)
      (*recHitsCPUEB)[i].setOutOfTimeAmplitude(sample, recHitsGPUEB.amplitudesAll[offset + sample]);
  }
  if (produceEE_) {
    auto const& recHitsGPUEE = event.get(recHitsGPUEE_);
    auto recHitsCPUEE = std::make_unique<EEUncalibratedRecHitCollection>();
    recHitsCPUEE->reserve(recHitsGPUEE.amplitude.size());
    for (uint32_t i = 0; i < recHitsGPUEE.amplitude.size(); ++i) {
      recHitsCPUEE->emplace_back(DetId{recHitsGPUEE.did[i]},
                                 recHitsGPUEE.amplitude[i],
                                 recHitsGPUEE.pedestal[i],
                                 recHitsGPUEE.jitter[i],
                                 recHitsGPUEE.chi2[i],
                                 recHitsGPUEE.flags[i]);
      (*recHitsCPUEB)[i].setAmplitudeError(recHitsGPUEE.amplitudeError[i]);
      (*recHitsCPUEE)[i].setJitterError(recHitsGPUEE.jitterError[i]);
      auto const offset = i * EcalDataFrame::MAXSAMPLES;
      for (uint32_t sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
        (*recHitsCPUEE)[i].setOutOfTimeAmplitude(sample, recHitsGPUEE.amplitudesAll[offset + sample]);
      }
    }
    event.put(std::move(recHitsCPUEE), recHitsLabelCPUEE_);
  }
  event.put(std::move(recHitsCPUEB), recHitsLabelCPUEB_);
}

DEFINE_FWK_MODULE(EcalUncalibRecHitConvertGPU2CPUFormat);