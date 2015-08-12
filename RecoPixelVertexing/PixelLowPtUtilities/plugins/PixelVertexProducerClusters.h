#ifndef PixelVertexProducerClusters_H
#define PixelVertexProducerClusters_H

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

namespace edm { class Run; class Event; class EventSetup; }

class TrackerGeometry;

class PixelVertexProducerClusters : public edm::global::EDProducer<>
{
public:
  explicit PixelVertexProducerClusters(const edm::ParameterSet& ps);
  ~PixelVertexProducerClusters();
  virtual void produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const override;

private:
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelToken_;
};
#endif
