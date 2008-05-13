#ifndef TrackTrajectorySateOnDetInfos_H
#define TrackTrajectorySateOnDetInfos_H

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include <vector>

namespace reco {

class TrajectorySateOnDetInfo {
public:
  typedef SiStripRecHit2D::ClusterRef ClusterRef;

  TrajectorySateOnDetInfo() {}
 ~TrajectorySateOnDetInfo() {}
  TrajectorySateOnDetInfo(const LocalTrajectoryParameters theLocalParameters, std::vector<float> theLocalErrors, const ClusterRef theCluster);

  unsigned int charge         ();
  double       thickness      (edm::ESHandle<TrackerGeometry> tkGeom);
  double       chargeOverPath (edm::ESHandle<TrackerGeometry> tkGeom);
  double       pathLength     (edm::ESHandle<TrackerGeometry> tkGeom);
  unsigned int clusterSize();



  LocalVector  momentum       ();
  LocalPoint   point          ();


private:
  LocalTrajectoryParameters          _theLocalParameters;
  std::vector<float>                 _theLocalErrors;
  ClusterRef                         _theCluster;
  
};


typedef std::vector<TrajectorySateOnDetInfo> TrajectorySateOnDetInfoCollection;

}
#endif
