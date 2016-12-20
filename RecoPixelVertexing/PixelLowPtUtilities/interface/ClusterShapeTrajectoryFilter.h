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

  virtual ~ClusterShapeTrajectoryFilter();

  void setEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  virtual bool qualityFilter(const TempTrajectory&) const;
  virtual bool qualityFilter(const Trajectory&) const;
 
  virtual bool toBeContinued(TempTrajectory&) const;
  virtual bool toBeContinued(Trajectory&) const;

  virtual std::string name() const { return "ClusterShapeTrajectoryFilter"; }

 private:
  edm::EDGetTokenT<SiPixelClusterShapeCache> theCacheToken;
  const SiPixelClusterShapeCache *theCache;
  const ClusterShapeHitFilter * theFilter;
};

#endif
