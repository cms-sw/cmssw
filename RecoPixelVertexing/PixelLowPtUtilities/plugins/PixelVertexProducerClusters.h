#ifndef PixelVertexProducerClusters_H
#define PixelVertexProducerClusters_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm { class Run; class Event; class EventSetup; }

class TrackerGeometry;
class VertexHit;

class PixelVertexProducerClusters : public edm::EDProducer
{
public:
  explicit PixelVertexProducerClusters(const edm::ParameterSet& ps);
  ~PixelVertexProducerClusters();
  int getContainedHits(std::vector<VertexHit> hits, float z0, float & chi);
  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;
 
private:
  void beginRun(edm::Run const & run, edm::EventSetup const & es) override;

  edm::ParameterSet theConfig;

  const TrackerGeometry* theTracker;
};
#endif
