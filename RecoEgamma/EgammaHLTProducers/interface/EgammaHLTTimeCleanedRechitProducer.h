#ifndef RecoEgamma_EgammayHLTProducers_EgammaHLTTimeCleanedRechitProducer_h_
#define RecoEgamma_EgammayHLTProducers_EgammaHLTTimeCleanedRechitProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EgammaHLTTimeCleanedRechitProducer : public edm::EDProducer {
  
 public:
  
  EgammaHLTTimeCleanedRechitProducer(const edm::ParameterSet& ps);
  ~EgammaHLTTimeCleanedRechitProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
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


