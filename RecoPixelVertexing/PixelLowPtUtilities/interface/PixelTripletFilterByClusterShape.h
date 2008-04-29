#ifndef _PixelTripletFilterByClusterShape_h_
#define _PixelTripletFilterByClusterShape_h_

#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include <vector>

#define MaxSize 20

class PixelTripletFilterByClusterShape 
{
  public:
    PixelTripletFilterByClusterShape(const edm::EventSetup& es);
    ~PixelTripletFilterByClusterShape();
    bool checkTrack(std::vector<const TrackingRecHit*> recHits,
                    std::vector<LocalVector> localDirs);

  private:
    void loadClusterLimits();
    bool isInside(double a[2][2], std::pair<double,double> movement);
    bool isCompatible(const SiPixelRecHit *recHit, const LocalVector& dir);
 
    const TrackerGeometry* theTracker;
 
    static bool isFirst;
    static double limits[2][MaxSize + 1][MaxSize + 1][2][2][2];
};

#endif

