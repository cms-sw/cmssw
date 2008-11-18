#ifndef TrackTrajectoryStateOnDetInfos_H
#define TrackTrajectoryStateOnDetInfos_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include <vector>

namespace reco {

class TrajectoryStateOnDetInfo {
public:
  TrajectoryStateOnDetInfo() {}
 ~TrajectoryStateOnDetInfo() {}
  TrajectoryStateOnDetInfo(const LocalTrajectoryParameters theLocalParameters, std::vector<float> theLocalErrors, const TrackingRecHitRef theTrackingRecHit);

  TrackingRecHitRef recHit         ();
  LocalVector       momentum       ();
  LocalPoint        point          ();


private:
  LocalTrajectoryParameters          _theLocalParameters;
  std::vector<float>                 _theLocalErrors;
  TrackingRecHitRef                  _theTrackingRecHit;
  
};


typedef std::vector<TrajectoryStateOnDetInfo> TrajectoryStateOnDetInfoCollection;

}
#endif
