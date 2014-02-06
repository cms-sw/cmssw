#ifndef _ValidHitPairFilter_h_
#define _ValidHitPairFilter_h_

#include <vector>

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

namespace edm { class ParameterSet; class EventSetup; }
class TrackingRecHit;
class Track;
class FreeTrajectoryState;
class TrackerGeometry;
class GeometricSearchTracker;
class MagneticField;
class Propagator;
class DetLayer;
class GeomDet;
class TrackerTopology;

class ValidHitPairFilter : public PixelTrackFilter 
{
public:
  ValidHitPairFilter(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
  virtual ~ValidHitPairFilter();
  void update(const edm::Event& ev, const edm::EventSetup& es) override;
  virtual bool operator()(const reco::Track * track,
                          const std::vector<const TrackingRecHit *>& recHits,
			  const TrackerTopology *tTopo) const;

private:
  int getLayer(const TrackingRecHit & recHit, const TrackerTopology *tTopo) const;
  std::vector<int> getMissingLayers(int a, int b) const;
  FreeTrajectoryState getTrajectory(const reco::Track & track) const;
  std::vector<const GeomDet *> getCloseDets
    (int il, float rz, const std::vector<float>& rzB,
     float ph, const std::vector<float>& phB, const TrackerTopology *tTopo) const;

  const TrackerGeometry * theTracker;
  const GeometricSearchTracker * theGSTracker;
  const MagneticField * theMagneticField;
  const Propagator*    thePropagator;

  const std::vector<DetLayer *> detLayers;
  std::vector<float> rzBounds[7];
  std::vector<float> phBounds[7];
};

#endif
