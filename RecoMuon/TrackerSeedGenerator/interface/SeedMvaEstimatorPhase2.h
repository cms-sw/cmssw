#ifndef RecoMuon_TrackerSeedGenerator_SeedMvaEstimatorPhase2_h
#define RecoMuon_TrackerSeedGenerator_SeedMvaEstimatorPhase2_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <memory>
#include <string>

typedef pair<const DetLayer*, TrajectoryStateOnSurface> LayerTSOS;
typedef pair<const DetLayer*, const TrackingRecHit*> LayerHit;

class GBRForest;

namespace edm {
  class FileInPath;
}

class SeedMvaEstimatorPhase2 {
public:
  SeedMvaEstimatorPhase2(const edm::FileInPath& weightsfile,
                         const std::vector<double>& scale_mean,
                         const std::vector<double>& scale_std);
  ~SeedMvaEstimatorPhase2();

  double computeMva(const TrajectorySeed&,
                    const GlobalVector&,
                    const GlobalPoint&,
                    const edm::Handle<l1t::TrackerMuonCollection>&,
                    const edm::ESHandle<MagneticField>&,
                    const Propagator&,
                    const GeometricSearchTracker&) const;

private:
  std::unique_ptr<const GBRForest> gbrForest_;
  const std::vector<double> scale_mean_;
  const std::vector<double> scale_std_;

  vector<LayerTSOS> getTsosOnPixels(const TTTrack<Ref_Phase2TrackerDigi_>&,
                                    const edm::ESHandle<MagneticField>&,
                                    const Propagator&,
                                    const GeometricSearchTracker&) const;

  vector<pair<LayerHit, LayerTSOS> > getHitTsosPairs(const TrajectorySeed&,
                                                     const edm::Handle<l1t::TrackerMuonCollection>&,
                                                     const edm::ESHandle<MagneticField>&,
                                                     const Propagator&,
                                                     const GeometricSearchTracker&) const;

  void getL1TTVariables(const TrajectorySeed&,
                        const GlobalVector&,
                        const GlobalPoint&,
                        const edm::Handle<l1t::TrackerMuonCollection>&,
                        float&,
                        float&) const;
  void getHitL1TkVariables(const TrajectorySeed&,
                           const edm::Handle<l1t::TrackerMuonCollection>&,
                           const edm::ESHandle<MagneticField>&,
                           const Propagator&,
                           const GeometricSearchTracker&,
                           float&,
                           float&,
                           float&) const;
};
#endif
