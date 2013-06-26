#ifndef DeDxTools_H
#define DeDxTools_H
#include <vector>
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

namespace DeDxTools  {
 
  struct RawHits {
    double charge;
    double angleCosine;
    DetId detId;
    const TrajectoryMeasurement* trajectoryMeasurement;
    int   NSaturating;
  };

  inline const SiStripCluster* GetCluster(const TrackerSingleRecHit * hit) { return &hit->stripCluster();}
  inline const SiStripCluster* GetCluster(const TrackerSingleRecHit & hit) {return &hit.stripCluster();}
  void   trajectoryRawHits(const edm::Ref<std::vector<Trajectory> >& trajectory, std::vector<RawHits>& hits, bool usePixel, bool useStrip);
  double genericAverage   (const reco::DeDxHitCollection &, float expo = 1.);
  bool shapeSelection(const std::vector<uint8_t> & ampls);
}

#endif
