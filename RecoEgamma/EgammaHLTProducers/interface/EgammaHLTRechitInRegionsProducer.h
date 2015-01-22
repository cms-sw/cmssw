#ifndef RecoEgamma_EgammayHLTProducers_EgammaHLTRechitInRegionsProducer_h_
#define RecoEgamma_EgammayHLTProducers_EgammaHLTRechitInRegionsProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EgammaHLTRechitInRegionsProducer : public edm::stream::EDProducer<> {
  
 public:
  
  EgammaHLTRechitInRegionsProducer(const edm::ParameterSet& ps);
  ~EgammaHLTRechitInRegionsProducer();

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  const bool useUncalib_;
  const edm::InputTag l1TagIsolated_;
  const edm::InputTag l1TagNonIsolated_;

  const bool doIsolated_;

  const double l1LowerThr_;
  const double l1UpperThr_;
  const double l1LowerThrIgnoreIsolation_;

  const double regionEtaMargin_;
  const double regionPhiMargin_;

  const std::vector<edm::InputTag> hitLabels;
  const std::vector<std::string> productLabels;

  std::vector<edm::EDGetTokenT<EcalRecHitCollection>> hitTokens;
  std::vector<edm::EDGetTokenT<EcalUncalibratedRecHitCollection>> uncalibHitTokens;
};


#endif


