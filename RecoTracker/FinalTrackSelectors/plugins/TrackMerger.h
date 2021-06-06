#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DuplicateTrackType.h"

class dso_hidden TrackMerger {
public:
  TrackMerger(const edm::ParameterSet &iConfig, edm::ConsumesCollector);
  ~TrackMerger();

  void init(const edm::EventSetup &iSetup);

  TrackCandidate merge(const reco::Track &inner, const reco::Track &outer, DuplicateTrackType duplicateType) const;

private:
  edm::ESHandle<TrackerGeometry> theGeometry;
  edm::ESHandle<MagneticField> theMagField;
  bool useInnermostState_;
  bool debug_;
  std::string theBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  edm::ESHandle<TrackerTopology> theTrkTopo;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> builderToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;

  void addSecondTrackHits(std::vector<const TrackingRecHit *> &hits, const reco::Track &outer) const;
  void sortByHitPosition(const GlobalVector &v,
                         const std::vector<const TrackingRecHit *> &hits,
                         TrackCandidate::RecHitContainer &ownHits) const;

  class GlobalMomentumSort {
  public:
    GlobalMomentumSort(const GlobalVector &dir) : dir_(dir) {}
    bool operator()(const TransientTrackingRecHit::RecHitPointer &hit1,
                    const TransientTrackingRecHit::RecHitPointer &hit2) const;

  private:
    GlobalVector dir_;
  };
  class MomentumSort {
  public:
    MomentumSort(const GlobalVector &dir, const TrackerGeometry *geometry) : dir_(dir), geom_(geometry) {}
    bool operator()(const TrackingRecHit *hit1, const TrackingRecHit *hit2) const;

  private:
    GlobalVector dir_;
    const TrackerGeometry *geom_;
  };
};
