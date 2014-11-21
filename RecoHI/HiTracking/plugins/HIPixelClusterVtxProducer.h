#ifndef HIPixelClusterVtxProducer_H
#define HIPixelClusterVtxProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

namespace edm { class Run; class Event; class EventSetup; }

class TrackerGeometry;

class HIPixelClusterVtxProducer : public edm::EDProducer
{
public:
  explicit HIPixelClusterVtxProducer(const edm::ParameterSet& ps);
  ~HIPixelClusterVtxProducer();
 
private:
  struct VertexHit
  {
    float z;
    float r;
    float w;
  };

  virtual void produce(edm::Event& ev, const edm::EventSetup& es);
  int getContainedHits(const std::vector<VertexHit> &hits, double z0, double &chi);

  std::string srcPixelsString_; //pixel rec hits
  edm::EDGetTokenT<SiPixelRecHitCollection> srcPixels_;

  double minZ_;
  double maxZ_;
  double zStep_;

};
#endif
