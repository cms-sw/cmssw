#ifndef PixelVertexProducerMedian_H
#define PixelVertexProducerMedian_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm { class Event; class EventSetup; }

class PixelVertexProducerMedian : public edm::EDProducer
{
public:
  explicit PixelVertexProducerMedian(const edm::ParameterSet& ps);
  ~PixelVertexProducerMedian();
  virtual void produce(edm::Event& ev, const edm::EventSetup& es);
 
private:
  edm::ParameterSet theConfig;
  double thePtMin;
};
#endif
