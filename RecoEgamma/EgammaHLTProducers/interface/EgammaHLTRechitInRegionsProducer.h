#ifndef RecoEgamma_EgammayHLTProducers_EgammaHLTRechitInRegionsProducer_h_
#define RecoEgamma_EgammayHLTProducers_EgammaHLTRechitInRegionsProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
//#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

class EgammaHLTRechitInRegionsProducer : public edm::EDProducer {
  
 public:
  
  EgammaHLTRechitInRegionsProducer(const edm::ParameterSet& ps);
  ~EgammaHLTRechitInRegionsProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  bool useUncalib_;
  bool doIsolated_;
  edm::InputTag hitproducer_;
  std::string hitcollection_;
  
  edm::InputTag l1TagIsolated_;
  edm::InputTag l1TagNonIsolated_;
  
  double l1LowerThr_;
  double l1UpperThr_;
  double l1LowerThrIgnoreIsolation_;
  
  double regionEtaMargin_;
  double regionPhiMargin_;
  
  std::vector<edm::InputTag> hitLabels;
  std::vector<std::string> productLabels;
  std::vector<edm::EDGetTokenT<EcalRecHitCollection>> hitTokens;
  std::vector<edm::EDGetTokenT<EcalUncalibratedRecHitCollection>> uncalibHitTokens;
};


#endif


