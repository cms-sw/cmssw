#ifndef PixelMatchElectronProducer_h
#define PixelMatchElectronProducer_h
  
//
// Package:         RecoEgamma/PixelMatchElectronProducers
// Class:           PixelMatchElectronProducer
// 
// Description:   
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
 
#include "DataFormats/Common/interface/EDProduct.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
  
class PixelMatchElectronAlgo;
 
class PixelMatchElectronProducer : public edm::EDProducer
{
 public:
  
  explicit PixelMatchElectronProducer(const edm::ParameterSet& conf);
  
  virtual ~PixelMatchElectronProducer();
  
  virtual void beginJob(edm::EventSetup const&iSetup);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
 private:
 
  const edm::ParameterSet conf_;
  
  PixelMatchElectronAlgo* algo_;  
 
};
  
#endif
 


