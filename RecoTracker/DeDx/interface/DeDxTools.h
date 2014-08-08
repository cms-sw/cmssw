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

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TFile.h"
#include "TChain.h"


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

  int getCharge(const SiStripCluster* cluster, int& nSatStrip, const GeomDetUnit& detUnit, const std::vector< std::vector< float > >& calibGains, const unsigned int& m_off );
  void makeCalibrationMap(const std::string& m_calibrationPath, const TrackerGeometry& tkGeom, std::vector< std::vector< float > >& calibGains, const unsigned int& m_off);

}

#endif
