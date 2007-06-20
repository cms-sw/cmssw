#ifndef _ClusterShapeTrackFilter_h_
#define _ClusterShapeTrackFilter_h_

#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include <vector>

using namespace std;

#define MaxSize 20

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackHitsFilter.h"

class ClusterShapeTrackFilter : public TrackHitsFilter 
{
  public:
    ClusterShapeTrackFilter
      (const edm::ParameterSet& ps);
//      (const edm::EventSetup& es);
    virtual ~ClusterShapeTrackFilter();
    virtual bool operator()(const reco::Track*, vector<const TrackingRecHit *> hits) const;

  private:
    void loadClusterLimits();
    bool isInside(const double a[2][2], pair<double,double> movement) const;
    bool isCompatible(const SiPixelRecHit *recHit, const LocalVector& dir) const;
 
    const TrackerGeometry* theTracker;
 
    /*static*/ double limits[2][MaxSize + 1][MaxSize + 1][2][2][2];
};

#endif

