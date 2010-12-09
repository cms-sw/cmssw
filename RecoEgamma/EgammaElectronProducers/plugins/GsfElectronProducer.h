#ifndef GsfElectronProducer_h
#define GsfElectronProducer_h

//
// Package:         RecoEgamma/EgammaElectronProducers
// Class:           GsfElectronProducer
//
// Description:


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GsfElectronAlgo;

class GsfElectronProducer : public edm::EDProducer
{
 public:

  explicit GsfElectronProducer(const edm::ParameterSet& conf);
  virtual ~GsfElectronProducer();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:

  GsfElectronAlgo* algo_;
};
#endif
