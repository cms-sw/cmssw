#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TripletFilter.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

using namespace std;

/*****************************************************************************/
TripletFilter::TripletFilter(const edm::EventSetup& es)
{
  // Get cluster shape hit filter
  edm::ESHandle<ClusterShapeHitFilter> shape;
  es.get<CkfComponentsRecord>().get("ClusterShapeHitFilter",shape);
  theFilter = shape.product();
}

/*****************************************************************************/
TripletFilter::~TripletFilter()
{
}

/*****************************************************************************/
bool TripletFilter::checkTrack
(const vector<const TrackingRecHit*>& recHits, const vector<LocalVector>& localDirs, const TrackerTopology *tTopo, const SiPixelClusterShapeCache& clusterShapeCache)
{
  bool ok = true;

  vector<LocalVector>::const_iterator localDir = localDirs.begin();
  for(vector<const TrackingRecHit*>::const_iterator recHit = recHits.begin();
                                                    recHit!= recHits.end();
                                                    recHit++)
  {
    const SiPixelRecHit* pixelRecHit =
      dynamic_cast<const SiPixelRecHit *>(*recHit);

    if(! pixelRecHit->isValid())
    {  ok = false; break; }

    if(! theFilter->isCompatible(*pixelRecHit, *localDir, clusterShapeCache))
    {
      LogTrace("MinBiasTracking")
       << "  [TripletFilter] clusShape problem"
       << HitInfo::getInfo(**recHit,tTopo);

      ok = false; break;
    }

    localDir++;
  }

  return ok;
}

/*****************************************************************************/
bool TripletFilter::checkTrack
(const vector<const TrackingRecHit*>& recHits, const vector<GlobalVector>& globalDirs, const TrackerTopology *tTopo, const SiPixelClusterShapeCache& clusterShapeCache)
{
  bool ok = true;

  vector<GlobalVector>::const_iterator globalDir = globalDirs.begin();
  for(vector<const TrackingRecHit*>::const_iterator recHit = recHits.begin();
                                                    recHit!= recHits.end();
                                                    recHit++)
  {
    const SiPixelRecHit* pixelRecHit =
      dynamic_cast<const SiPixelRecHit *>(*recHit);

    if(! pixelRecHit->isValid())
    {  ok = false; break; }

    if(! theFilter->isCompatible(*pixelRecHit, *globalDir, clusterShapeCache))
    {
      LogTrace("MinBiasTracking")
       << "  [TripletFilter] clusShape problem"
       << HitInfo::getInfo(**recHit,tTopo);

      ok = false; break;
    }

    globalDir++;
  }

  return ok;
}

