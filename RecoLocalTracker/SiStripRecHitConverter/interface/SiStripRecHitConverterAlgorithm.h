#ifndef SiStripRecHitConverterAlgorithm_h
#define SiStripRecHitConverterAlgorithm_h

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include <memory>

namespace edm {
  class ConsumesCollector;
  class ParameterSet;
  class ParameterSetDescription;
  class EventSetup;
}  // namespace edm
class TrackerDigiGeometryRecord;
class TkStripCPERecord;
class SiStripQualityRcd;

class SiStripRecHitConverterAlgorithm {
public:
  struct products {
    std::unique_ptr<SiStripMatchedRecHit2DCollection> matched;
    std::unique_ptr<SiStripRecHit2DCollection> rphi, stereo, rphiUnmatched, stereoUnmatched;
    products()
        : matched(new SiStripMatchedRecHit2DCollection),
          rphi(new SiStripRecHit2DCollection),
          stereo(new SiStripRecHit2DCollection),
          rphiUnmatched(new SiStripRecHit2DCollection),
          stereoUnmatched(new SiStripRecHit2DCollection) {}

    void shrink_to_fit() {
      matched->shrink_to_fit();
      rphi->shrink_to_fit();
      stereo->shrink_to_fit();
      rphiUnmatched->shrink_to_fit();
      stereoUnmatched->shrink_to_fit();
    }
  };

  SiStripRecHitConverterAlgorithm(const edm::ParameterSet&, edm::ConsumesCollector);
  void initialize(const edm::EventSetup&);
  SiStripRecHitConverterAlgorithm initializedClone(const edm::EventSetup&) const;
  template <typename... ExtraFillers>
  void run(edm::Handle<edmNew::DetSetVector<SiStripCluster>> input, products& output, ExtraFillers&&... extra_fillers);
  template <typename... ExtraFillers>
  void run(edm::Handle<edmNew::DetSetVector<SiStripCluster>> input,
           products& output,
           LocalVector trackdirection,
           ExtraFillers&&... extra_fillers);

  static void fillPSetDescription(edm::ParameterSetDescription& desc);

private:
  void match(products& output, LocalVector trackdirection) const;
  void fillBad128StripBlocks(const uint32_t detid, bool bad128StripBlocks[6]) const;
  bool isMasked(const SiStripCluster& cluster, bool bad128StripBlocks[6]) const;
  bool useModule(const uint32_t id) const;

  bool useQuality, maskBad128StripBlocks, doMatching;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerToken;
  edm::ESGetToken<StripClusterParameterEstimator, TkStripCPERecord> cpeToken;
  edm::ESGetToken<SiStripRecHitMatcher, TkStripCPERecord> matcherToken;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken;
  const TrackerGeometry* tracker = nullptr;
  const StripClusterParameterEstimator* parameterestimator = nullptr;
  const SiStripRecHitMatcher* matcher = nullptr;
  const SiStripQuality* quality = nullptr;

  typedef SiStripRecHit2DCollection::FastFiller Collector;
};

template <typename... ExtraFillers>
void SiStripRecHitConverterAlgorithm::run(edm::Handle<edmNew::DetSetVector<SiStripCluster>> input,
                                          products& output,
                                          ExtraFillers&&... extra_fillers) {
  run(input, output, LocalVector(0., 0., 0.), std::forward<ExtraFillers>(extra_fillers)...);
}

template <typename... ExtraFillers>
void SiStripRecHitConverterAlgorithm::run(edm::Handle<edmNew::DetSetVector<SiStripCluster>> inputhandle,
                                          products& output,
                                          LocalVector trackdirection,
                                          ExtraFillers&&... extra_fillers) {
  auto const inputID = inputhandle.id();
  unsigned int nIDs[2]{};
  unsigned int nCs[2]{};
  for (auto const& DS : *inputhandle) {
    auto id = DS.id();
    if (!useModule(id))
      continue;

    unsigned int iStereo = StripSubdetector(id).stereo();
    nIDs[iStereo]++;

    bool bad128StripBlocks[6];
    fillBad128StripBlocks(id, bad128StripBlocks);

    for (auto const& cluster : DS) {
      if (isMasked(cluster, bad128StripBlocks))
        continue;

      nCs[iStereo]++;
    }
  }
  output.rphi->reserve(nIDs[0], nCs[0]);
  output.stereo->reserve(nIDs[1], nCs[1]);

  for (auto const& DS : *inputhandle) {
    auto id = DS.id();
    if (!useModule(id))
      continue;

    Collector collector = StripSubdetector(id).stereo() ? Collector(*output.stereo, id) : Collector(*output.rphi, id);

    bool bad128StripBlocks[6];
    fillBad128StripBlocks(id, bad128StripBlocks);

    GeomDetUnit const& du = *(tracker->idToDetUnit(id));
    for (auto const& cluster : DS) {
      if (isMasked(cluster, bad128StripBlocks))
        continue;

      StripClusterParameterEstimator::LocalValues parameters = parameterestimator->localParameters(cluster, du);
      OmniClusterRef clusterRef(inputID, &cluster, DS.makeKeyOf(&cluster));
      collector.push_back(SiStripRecHit2D(parameters.first, parameters.second, du, clusterRef));

      (..., extra_fillers.fill(*inputhandle, clusterRef.index(), collector.back(), id, du));
    }

    if (collector.empty())
      collector.abort();
  }
  if (doMatching) {
    match(output, trackdirection);
  }
}

inline bool SiStripRecHitConverterAlgorithm::isMasked(const SiStripCluster& cluster, bool bad128StripBlocks[6]) const {
  if (maskBad128StripBlocks) {
    if (bad128StripBlocks[cluster.firstStrip() >> 7]) {
      if (bad128StripBlocks[(cluster.firstStrip() + cluster.amplitudes().size()) >> 7] ||
          bad128StripBlocks[static_cast<int32_t>(cluster.barycenter() - 0.499999) >> 7]) {
        return true;
      }
    } else {
      if (bad128StripBlocks[(cluster.firstStrip() + cluster.amplitudes().size()) >> 7] &&
          bad128StripBlocks[static_cast<int32_t>(cluster.barycenter() - 0.499999) >> 7]) {
        return true;
      }
    }
  }
  return false;
}

inline bool SiStripRecHitConverterAlgorithm::useModule(const uint32_t id) const {
  const StripGeomDetUnit* stripdet = (const StripGeomDetUnit*)tracker->idToDetUnit(id);
  if (stripdet == nullptr)
    edm::LogWarning("SiStripRecHitConverter") << "Detid=" << id << " not found";
  return stripdet != nullptr && (!useQuality || quality->IsModuleUsable(id));
}

#endif
