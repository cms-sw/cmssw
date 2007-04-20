#ifndef PixelMatchGsfElectronProducer_h
#define PixelMatchGsfElectronProducer_h
  
//
// Package:         RecoEgamma/EgammaElectronProducers
// Class:           PixelMatchGsfElectronProducer
// 
// Description:   
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class PixelMatchElectronAlgo;

class PixelMatchGsfElectronProducer : public edm::EDProducer
{
 public:

  explicit PixelMatchGsfElectronProducer(const edm::ParameterSet& conf);

  virtual ~PixelMatchGsfElectronProducer();

  virtual void beginJob(edm::EventSetup const&iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:

  const edm::ParameterSet conf_;

  PixelMatchElectronAlgo* algo_;
  std::string  seedProducer_;
};
#endif
