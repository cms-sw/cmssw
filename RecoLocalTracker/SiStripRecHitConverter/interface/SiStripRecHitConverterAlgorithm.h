#ifndef SiStripRecHitConverterAlgorithm_h
#define SiStripRecHitConverterAlgorithm_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

namespace edm {
   class ParameterSet;
   template<typename T> class RefGetter;
   class EventSetup;
}

class SiStripRecHitConverterAlgorithm 
{

 public:
  
  struct products {
    std::auto_ptr<SiStripMatchedRecHit2DCollection> matched;
    std::auto_ptr<SiStripRecHit2DCollection> rphi, stereo, rphiUnmatched, stereoUnmatched;
    products() 
      :  matched(new SiStripMatchedRecHit2DCollection),
	 rphi(new SiStripRecHit2DCollection),
	 stereo(new SiStripRecHit2DCollection),
	 rphiUnmatched(new SiStripRecHit2DCollection),
	 stereoUnmatched(new SiStripRecHit2DCollection) {}
  };
  
  SiStripRecHitConverterAlgorithm(const edm::ParameterSet&);
  void initialize(const edm::EventSetup&);  
  void run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > input, products& output);
  void run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > input, products& output, LocalVector trackdirection);
  
 private:
  
  void match(products& output, LocalVector trackdirection) const;
  void fillBad128StripBlocks(const uint32_t detid, bool bad128StripBlocks[6]) const;
  bool isMasked(const SiStripCluster &cluster, bool bad128StripBlocks[6]) const;
  bool useModule(const uint32_t id) const;
  
  bool useQuality, maskBad128StripBlocks;
  uint32_t tracker_cache_id, cpe_cache_id, quality_cache_id;
  edm::ESInputTag cpeTag, matcherTag, qualityTag;
  edm::ESHandle<TrackerGeometry> tracker;
  edm::ESHandle<StripClusterParameterEstimator> parameterestimator;
  edm::ESHandle<SiStripRecHitMatcher> matcher;
  edm::ESHandle<SiStripQuality> quality;
  
  typedef SiStripRecHit2DCollection::FastFiller Collector;

};

#endif
