#ifndef ElectronSqPtTkIsolationProducer_h
#define ElectronSqPtTkIsolationProducer_h



#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ElectronSqPtTkIsolationProducer : public edm::EDProducer {
 public:
  explicit ElectronSqPtTkIsolationProducer(const edm::ParameterSet&);
  ~ElectronSqPtTkIsolationProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag electronProducer_;
  edm::InputTag trackProducer_;

  double ptMin_;
  double intRadius_;
  double extRadius_;
  double maxVtxDist_;

  bool absolut_;
  
  edm::ParameterSet conf_;

};


#endif
