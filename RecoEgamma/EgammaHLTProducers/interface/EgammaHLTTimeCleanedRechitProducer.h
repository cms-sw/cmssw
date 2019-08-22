#ifndef RecoEgamma_EgammayHLTProducers_EgammaHLTTimeCleanedRechitProducer_h_
#define RecoEgamma_EgammayHLTProducers_EgammaHLTTimeCleanedRechitProducer_h_

#include <memory>

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EgammaHLTTimeCleanedRechitProducer : public edm::global::EDProducer<> {
public:
  EgammaHLTTimeCleanedRechitProducer(const edm::ParameterSet& ps);
  ~EgammaHLTTimeCleanedRechitProducer() override;

  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  double timeMax_;
  double timeMin_;
  std::vector<edm::InputTag> hitLabels;
  std::vector<std::string> productLabels;
  std::vector<edm::EDGetTokenT<EcalRecHitCollection>> hitTokens;
  std::vector<edm::EDGetTokenT<EcalUncalibratedRecHitCollection>> uncalibHitTokens;
};

#endif
