#ifndef DeDxTools_H
#define DeDxTools_H
#include <vector>
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

namespace DeDxTools  {
 
  struct RawHits {
    double charge;
    double angleCosine;
    DetId detId;
    const TrajectoryMeasurement* trajectoryMeasurement;
    int   NSaturating;
  };

   void   trajectoryRawHits(const edm::Ref<std::vector<Trajectory> >& trajectory, std::vector<RawHits>& hits, bool usePixel, bool useStrip);
   double genericAverage   (const reco::DeDxHitCollection &, float expo = 1.);
}

#endif
