#ifndef SiStripRecHitConverterAlgorithm_h
#define SiStripRecHitConverterAlgorithm_h

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

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
  void run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > input, products& output);
  void run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > input, products& output, LocalVector trackdirection);

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

#endif
