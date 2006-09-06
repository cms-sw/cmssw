#ifndef _PixelVZeroProducer_h_
#define _PixelVZeroProducer_h_

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PixelVZeroProducer :  public edm::EDProducer {

public:
  explicit PixelVZeroProducer(const edm::ParameterSet& pset);

  ~PixelVZeroProducer();

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  edm::ParameterSet pset_;

  float minImpactPositiveDaughter,
        minImpactNegativeDaughter;
};
#endif

