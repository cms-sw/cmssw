#ifndef _ClusterShapeTrajectoryFilter_h_
#define _ClusterShapeTrajectoryFilter_h_

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm { class ParameterSet; class EventSetup; }

class SiPixelRecHit;
class SiStripRecHit2D;
class GlobalTrackingGeometry;
class MagneticField;
class SiPixelLorentzAngle;
class SiStripLorentzAngle;
class ClusterShapeHitFilter;
class SiPixelClusterShapeCache;

class ClusterShapeTrajectoryFilter : public TrajectoryFilter {
 public:
  ClusterShapeTrajectoryFilter(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC);

  ~ClusterShapeTrajectoryFilter() override;

  void setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  bool qualityFilter(const TempTrajectory&) const override;
  bool qualityFilter(const Trajectory&) const override;
 
  bool toBeContinued(TempTrajectory&) const override;
  bool toBeContinued(Trajectory&) const override;

  std::string name() const override { return "ClusterShapeTrajectoryFilter"; }

 private:
  edm::EDGetTokenT<SiPixelClusterShapeCache> theCacheToken;
  const SiPixelClusterShapeCache *theCache;
  const ClusterShapeHitFilter * theFilter;
};

#endif
