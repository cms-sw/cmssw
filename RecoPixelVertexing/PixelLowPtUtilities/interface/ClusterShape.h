#ifndef _ClusterShape_h_
#define _ClusterShape_h_

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterData.h"

#include <vector>
#include <utility>

#define MaxSize 20

class ClusterShape
{
 public:
   ClusterShape();
   ~ClusterShape();
   void getExtra(const PixelGeomDetUnit& pixelDet,
                 const SiPixelRecHit& recHit, ClusterData& data);

 private:
   int getDirection(int low,int hig, int olow,int ohig);
   bool processColumn(std::pair<int,int> pos, bool inTheLoop);
   void determineShape
     (const PixelGeomDetUnit& pixelDet,
      const SiPixelRecHit& recHit,     ClusterData& data);
   void getOrientation
     (const PixelGeomDetUnit& pixelDet,ClusterData& data);

   int x[2],y[2], low,hig, olow,ohig, odir;
};

#endif

