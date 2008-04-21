#ifndef GlobalGsfElectronProducer_h
#define GlobalGsfElectronProducer_h
  
//
// Package:         RecoEgamma/EgammaElectronProducers
// Class:           GlobalGsfElectronProducer
// 
// Description:   
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GlobalGsfElectronAlgo;

class GlobalGsfElectronProducer : public edm::EDProducer
{
 public:

  explicit GlobalGsfElectronProducer(const edm::ParameterSet& conf);

  virtual ~GlobalGsfElectronProducer();

  virtual void beginJob(edm::EventSetup const&iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:

  GlobalGsfElectronAlgo* algo_;
};
#endif
