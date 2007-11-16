#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducerWithVertices_H 
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducerWithVertices_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class GlobalTrackingRegionProducerWithVertices : public TrackingRegionProducer
{
public:

  GlobalTrackingRegionProducerWithVertices(const edm::ParameterSet& cfg);
  virtual ~GlobalTrackingRegionProducerWithVertices() {}

  virtual std::vector<TrackingRegion* > regions
    (const edm::Event& ev, const edm::EventSetup&) const;

private:
  double thePtMin; 
  double theOriginRadius; 
  double theOriginHalfLength; 
  double theOriginZPos;
  bool thePrecise;
  
  bool theUseFoundVertices;
};

#endif 
