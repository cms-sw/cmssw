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

  int nMaxPrintout_; // max # of printouts
  int nEvt_;         // internal counter of events
  
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
  
  //HybridClusterAlgo * hybrid_p; // clustering algorithm
  //PositionCalc posCalculator_; // position calculation algorithm
  
  bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0));}
};


#endif


