#include "RecoPixelVertexing/PixelLowPtUtilities/plugins/GlobalTrackingRegionProducerWithVertices.h"

/*****************************************************************************/
GlobalTrackingRegionProducerWithVertices::GlobalTrackingRegionProducerWithVertices(const edm::ParameterSet& cfg)
{ 
  edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

  thePtMin            = regionPSet.getParameter<double>("ptMin");
  theOriginRadius     = regionPSet.getParameter<double>("originRadius");
  theOriginHalfLength = regionPSet.getParameter<double>("originHalfLength");
  theOriginZPos       = regionPSet.getParameter<double>("originZPos");
  thePrecise          = regionPSet.getParameter<bool>("precise"); 

  theUseFoundVertices = regionPSet.getParameter<bool>("useFoundVertices");
}   

/*****************************************************************************/
std::vector<TrackingRegion* > GlobalTrackingRegionProducerWithVertices::regions
    (const edm::Event& ev, const edm::EventSetup&) const
{
  std::vector<TrackingRegion* > result;

  if(theUseFoundVertices)
  {
    // Get originZPos from list of vertices (first or all)
    edm::Handle<reco::VertexCollection> vertexCollection;
    ev.getByType(vertexCollection);
    const reco::VertexCollection* vertices = vertexCollection.product();

      if(vertices->size() > 0)
  {
      std::cerr << " [TrackProducer] using vertex at z="
                << vertices->front().position().z() << " cm with "
                << vertices->front().tracksSize() << " tracks" << std::endl;
      double theOriginZPos_       = vertices->front().position().z();

      result.push_back(
        new GlobalTrackingRegion(thePtMin, theOriginRadius,
            theOriginHalfLength, theOriginZPos_, thePrecise) );
    }
    else
    {
      float theOriginHalfLength_ = 15.9;

      std::cerr << " [TrackProducer] no vertices found" << std::endl;
      result.push_back(
        new GlobalTrackingRegion(thePtMin, theOriginRadius,
            theOriginHalfLength_, theOriginZPos, thePrecise) );
    }
  }
  else
  {
    float theOriginHalfLength_ = 15.9;
    result.push_back(
      new GlobalTrackingRegion(thePtMin, theOriginRadius,
          theOriginHalfLength_, theOriginZPos, thePrecise) );
  }

  return result;
}
