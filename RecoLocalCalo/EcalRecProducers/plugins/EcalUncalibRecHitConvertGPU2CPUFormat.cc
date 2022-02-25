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
  edm::EDGetTokenT<InputProduct> recHitsGPUEB_;
  edm::EDGetTokenT<InputProduct> recHitsGPUEE_;

  std::string recHitsLabelCPUEB_, recHitsLabelCPUEE_;
};

void EcalUncalibRecHitConvertGPU2CPUFormat::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsLabelGPUEB", edm::InputTag("ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEB"));
 
  desc.add<std::string>("recHitsLabelCPUEB", "EcalUncalibRecHitsEB");
  
  desc.add<bool>("produceEE", false);

  // if(produceEE_)
  // {
  desc.add<edm::InputTag>("recHitsLabelGPUEE", edm::InputTag("ecalUncalibRecHitProducerGPU", "EcalUncalibRecHitsEE"));
  desc.add<std::string>("recHitsLabelCPUEE", "EcalUncalibRecHitsEE");
  // }


  confDesc.add("ecalUncalibRecHitConvertGPU2CPUFormat", desc);
}

EcalUncalibRecHitConvertGPU2CPUFormat::EcalUncalibRecHitConvertGPU2CPUFormat(const edm::ParameterSet& ps)
    : produceEE_{ps.getParameter<bool>("produceEE")},
      recHitsGPUEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsLabelGPUEB"))},
      recHitsLabelCPUEB_{ps.getParameter<std::string>("recHitsLabelCPUEB")} {
  produces<EBUncalibratedRecHitCollection>(recHitsLabelCPUEB_);
  if(produceEE_){
  recHitsGPUEE_ = consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsLabelGPUEE"));
  recHitsLabelCPUEE_ = ps.getParameter<std::string>("recHitsLabelCPUEE"); 
  produces<EEUncalibratedRecHitCollection>(recHitsLabelCPUEE_);
  }
}

EcalUncalibRecHitConvertGPU2CPUFormat::~EcalUncalibRecHitConvertGPU2CPUFormat() {}

void EcalUncalibRecHitConvertGPU2CPUFormat::produce(edm::Event& event, edm::EventSetup const& setup) {
  edm::Handle<InputProduct> hRecHitsGPUEB, hRecHitsGPUEE;
  
  event.getByToken(recHitsGPUEB_, hRecHitsGPUEB);
  auto recHitsCPUEB = std::make_unique<EBUncalibratedRecHitCollection>();
  recHitsCPUEB->reserve(hRecHitsGPUEB->amplitude.size());

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

  if(produceEE_){  
  event.getByToken(recHitsGPUEE_, hRecHitsGPUEE);
  auto recHitsCPUEE = std::make_unique<EEUncalibratedRecHitCollection>();
  recHitsCPUEE->reserve(hRecHitsGPUEE->amplitude.size());
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
  {
     (*recHitsCPUEE)[i].setOutOfTimeAmplitude(sample, hRecHitsGPUEE->amplitudesAll[offset + sample]);
  }
  
  
  }
  event.put(std::move(recHitsCPUEE), recHitsLabelCPUEE_);
  }
  event.put(std::move(recHitsCPUEB), recHitsLabelCPUEB_);
}

DEFINE_FWK_MODULE(EcalUncalibRecHitConvertGPU2CPUFormat);
