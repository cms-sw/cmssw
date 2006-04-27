#ifndef PixelTrackProducer_H
#define PixelTrackProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PixelTrackProducer :  public edm::EDProducer {

public:
  explicit PixelTrackProducer(const edm::ParameterSet& conf);

  ~PixelTrackProducer();

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);

private:

  edm::ParameterSet theConfig; 
};
#endif
