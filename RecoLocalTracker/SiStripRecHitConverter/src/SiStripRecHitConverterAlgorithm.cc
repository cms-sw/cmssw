#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/Ref.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(SiStripRecHitMatcher);


SiStripRecHitConverterAlgorithm::SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf) : 
  useQuality(conf.getParameter<bool>("useSiStripQuality")),
  maskBad128StripBlocks( conf.existsAs<bool>("MaskBadAPVFibers") && conf.getParameter<bool>("MaskBadAPVFibers")),
  tracker_cache_id(0),
  cpe_cache_id(0),
  quality_cache_id(0),
  cpeTag(conf.getParameter<edm::ESInputTag>("StripCPE")),
  matcherTag(conf.getParameter<edm::ESInputTag>("Matcher")),
  qualityTag(conf.getParameter<edm::ESInputTag>("siStripQualityLabel"))
{}

void SiStripRecHitConverterAlgorithm::
initialize(const edm::EventSetup& es) 
{
  uint32_t tk_cache_id = es.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  uint32_t c_cache_id = es.get<TkStripCPERecord>().cacheIdentifier();
  uint32_t q_cache_id = es.get<SiStripQualityRcd>().cacheIdentifier();

  if(tk_cache_id != tracker_cache_id) {
    es.get<TrackerDigiGeometryRecord>().get(tracker);
    tracker_cache_id = tk_cache_id;
  }
  if(c_cache_id != cpe_cache_id) {
    es.get<TkStripCPERecord>().get(matcherTag, matcher);
    es.get<TkStripCPERecord>().get(cpeTag, parameterestimator); 
    cpe_cache_id = c_cache_id;
  }
  if( useQuality && q_cache_id!=quality_cache_id) {
    es.get<SiStripQualityRcd>().get(qualityTag, quality);
    quality_cache_id = q_cache_id;
  }
}

void SiStripRecHitConverterAlgorithm::
run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > input, products& output) 
{ run(input, output, LocalVector(0.,0.,0.)); }



void SiStripRecHitConverterAlgorithm::
run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > inputhandle, products& output, LocalVector trackdirection)
{

  edmNew::DetSetVector<SiStripCluster>::const_iterator dse = inputhandle->end();
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator 
	 DS = inputhandle->begin(); DS != dse; ++DS ) {     
    edmNew::det_id_type id = (*DS).id();
    if(!useModule(id)) continue;

    Collector collector = StripSubdetector(id).stereo()  
      ? Collector(*output.stereo, id) 
      : Collector(*output.rphi,   id);

    bool bad128StripBlocks[6]; fillBad128StripBlocks( id, bad128StripBlocks);
    
    GeomDetUnit const & du = *(tracker->idToDetUnit(id));
    edmNew::DetSet<SiStripCluster>::const_iterator cle = (*DS).end();
    for(edmNew::DetSet<SiStripCluster>::const_iterator 
	  cluster = (*DS).begin();  cluster != cle; ++cluster ) {     

      if(isMasked(*cluster,bad128StripBlocks)) continue;

      StripClusterParameterEstimator::LocalValues parameters = 	parameterestimator->localParameters(*cluster,du);
      collector.push_back(SiStripRecHit2D( parameters.first, parameters.second, du, edmNew::makeRefTo(inputhandle,cluster) ));
    }

    if (collector.empty()) collector.abort();
  }
  match(output,trackdirection);
}


namespace {

  struct CollectorHelper {
    size_t nmatch;
    
    
    typedef edm::OwnVector<SiStripMatchedRecHit2D> CollectorMatched;
    typedef SiStripMatchedRecHit2DCollection::FastFiller Collector;
    
    Collector & m_collector;
    CollectorMatched & m_collectorMatched;
    SiStripRecHit2DCollection::FastFiller & m_fillerRphiUnm;
    std::vector<SiStripRecHit2D::ClusterRef::key_type>         & m_matchedSteroClusters;
    
    static inline SiStripRecHit2D const & stereoHit(edmNew::DetSet<SiStripRecHit2D>::const_iterator iter) {
      return *iter;
    }
    
    static inline SiStripRecHit2D const & monoHit(edmNew::DetSet<SiStripRecHit2D>::const_iterator iter) {
      return *iter;
    }
    
    struct Add {
      Add(CollectorHelper& ih) : h(ih){}
      CollectorHelper& h;
      void operator()(SiStripMatchedRecHit2D const & rh) { h.m_collectorMatched.push_back(rh);}
    };

    CollectorHelper & collector() {
      return *this;
    }

   void operator()(SiStripMatchedRecHit2D const & rh) {m_collectorMatched.push_back(rh);}


    CollectorHelper(
		    Collector & i_collector,
		    CollectorMatched & i_collectorMatched,
		    SiStripRecHit2DCollection::FastFiller & i_fillerRphiUnm,
		    std::vector<SiStripRecHit2D::ClusterRef::key_type> & i_matchedSteroClusters
		    ) : nmatch(0), 
			m_collector(i_collector),
			m_collectorMatched(i_collectorMatched),
			m_fillerRphiUnm(i_fillerRphiUnm),
			m_matchedSteroClusters(i_matchedSteroClusters)    {}
    
    void closure(edmNew::DetSet<SiStripRecHit2D>::const_iterator it) {
      if (!m_collectorMatched.empty()){
	nmatch+=m_collectorMatched.size();
	for (edm::OwnVector<SiStripMatchedRecHit2D>::const_iterator itm = m_collectorMatched.begin(),
	       edm = m_collectorMatched.end();
	     itm != edm; 
	     ++itm) {
	  m_collector.push_back(*itm);
	  // mark the stereo hit cluster as used, so that the hit won't go in the unmatched stereo ones
	    m_matchedSteroClusters.push_back(itm->stereoClusterRef().key()); 
	}
	m_collectorMatched.clear();
      } else {
	// store a copy of this rphi hit as an unmatched rphi hit
	m_fillerRphiUnm.push_back(*it);
      }
    }
  };
}


void SiStripRecHitConverterAlgorithm::
match(products& output, LocalVector trackdirection) const 
{
  int nmatch=0;
  edm::OwnVector<SiStripMatchedRecHit2D> collectorMatched; // gp/FIXME: avoid this
  
  // Remember the ends of the collections, as we will use them a lot
  SiStripRecHit2DCollection::const_iterator edStereoDet = output.stereo->end();
  SiStripRecHit2DCollection::const_iterator edRPhiDet   = output.rphi->end();
  
  // two work vectors for bookeeping clusters used by the stereo part of the matched hits
  std::vector<SiStripRecHit2D::ClusterRef::key_type>         matchedSteroClusters;
  
  for (SiStripRecHit2DCollection::const_iterator itRPhiDet = output.rphi->begin(); itRPhiDet != edRPhiDet; ++itRPhiDet) {
    edmNew::DetSet<SiStripRecHit2D> rphiHits = *itRPhiDet;
    StripSubdetector specDetId(rphiHits.detId());
    uint32_t partnerId = specDetId.partnerDetId();
    
    // if not part of a glued pair
    if (partnerId == 0) { 
      // I must copy these as unmatched 
      if (!rphiHits.empty()) {
	SiStripRecHit2DCollection::FastFiller filler(*output.rphiUnmatched, rphiHits.detId());
	filler.resize(rphiHits.size());
	std::copy(rphiHits.begin(), rphiHits.end(), filler.begin());
      }
      continue;
    }
    
    SiStripRecHit2DCollection::const_iterator itStereoDet = output.stereo->find(partnerId);
    
    // if the partner is not found (which probably can happen if it's empty)
    if (itStereoDet == edStereoDet) {
      // I must copy these as unmatched 
      if (!rphiHits.empty()) {
	SiStripRecHit2DCollection::FastFiller filler(*output.rphiUnmatched, rphiHits.detId());
	filler.resize(rphiHits.size());
	std::copy(rphiHits.begin(), rphiHits.end(), filler.begin());
      }
      continue;
    }
    
    edmNew::DetSet<SiStripRecHit2D> stereoHits = *itStereoDet;
    
     
    // Get ready for making glued hits
    const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker->idToDet(DetId(specDetId.glued()));
    typedef SiStripMatchedRecHit2DCollection::FastFiller Collector;
    Collector collector(*output.matched, specDetId.glued());
    
    // Prepare also the list for unmatched rphi hits
    SiStripRecHit2DCollection::FastFiller fillerRphiUnm(*output.rphiUnmatched, rphiHits.detId());
    
    // a list of clusters used by the matched part of the stereo hits in this detector
    matchedSteroClusters.clear();          // at the beginning, empty

#ifdef DOUBLE_MATCH
    CollectorHelper chelper(collector, collectorMatched,
		    fillerRphiUnm,
		    matchedSteroClusters
		    );
    matcher->doubleMatch(rphiHits.begin(), rphiHits.end(), 
			 stereoHits.begin(),stereoHits.end(),gluedDet,trackdirection,chelper);
    nmatch+=chelper.nmatch;
#else 
   // Make simple collection of this (gp:FIXME: why do we need it?)
    SiStripRecHitMatcher::SimpleHitCollection stereoSimpleHits;
    // gp:FIXME: use std::transform 
    stereoSimpleHits.reserve(stereoHits.size());
    for (edmNew::DetSet<SiStripRecHit2D>::const_iterator it = stereoHits.begin(), ed = stereoHits.end(); it != ed; ++it) {
      stereoSimpleHits.push_back(&*it);
    }

    for (edmNew::DetSet<SiStripRecHit2D>::const_iterator it = rphiHits.begin(), ed = rphiHits.end(); it != ed; ++it) {
      matcher->match(&(*it),stereoSimpleHits.begin(),stereoSimpleHits.end(),collectorMatched,gluedDet,trackdirection);
      if (collectorMatched.size()>0){
	nmatch+=collectorMatched.size();
	for (edm::OwnVector<SiStripMatchedRecHit2D>::const_iterator itm = collectorMatched.begin(),
	       edm = collectorMatched.end();
	     itm != edm; 
	     ++itm) {
	  collector.push_back(*itm);
	  // mark the stereo hit cluster as used, so that the hit won't go in the unmatched stereo ones
	    matchedSteroClusters.push_back(itm->stereoClusterRef().key());
	}
	collectorMatched.clear();
      } else {
	// store a copy of this rphi hit as an unmatched rphi hit
	fillerRphiUnm.push_back(*it);
      }
    }
    
#endif


    // discard matched hits if the collection is empty
    if (collector.empty()) collector.abort();
    
    // discard unmatched rphi hits if there are none
    if (fillerRphiUnm.empty()) fillerRphiUnm.abort();
    
    // now look for unmatched stereo hits    
    SiStripRecHit2DCollection::FastFiller fillerStereoUnm(*output.stereoUnmatched, stereoHits.detId());
      std::sort(matchedSteroClusters.begin(), matchedSteroClusters.end());
      for (edmNew::DetSet<SiStripRecHit2D>::const_iterator it = stereoHits.begin(), ed = stereoHits.end(); it != ed; ++it) {
	if (!std::binary_search(matchedSteroClusters.begin(), matchedSteroClusters.end(), it->cluster().key())) {
	  fillerStereoUnm.push_back(*it);
	}
      }
    if (fillerStereoUnm.empty()) fillerStereoUnm.abort(); 
    
    
  }
  
  for (SiStripRecHit2DCollection::const_iterator itStereoDet = output.stereo->begin(); itStereoDet != edStereoDet; ++itStereoDet) {
    edmNew::DetSet<SiStripRecHit2D> stereoHits = *itStereoDet;
    StripSubdetector specDetId(stereoHits.detId());
    uint32_t partnerId = specDetId.partnerDetId();
    if (partnerId == 0) continue;
    SiStripRecHit2DCollection::const_iterator itRPhiDet = output.rphi->find(partnerId);
    if (itRPhiDet == edRPhiDet) {
      if (!stereoHits.empty()) {
	SiStripRecHit2DCollection::FastFiller filler(*output.stereoUnmatched, stereoHits.detId());
	filler.resize(stereoHits.size());
	std::copy(stereoHits.begin(), stereoHits.end(), filler.begin());
      }
    }
  }
  
  edm::LogInfo("SiStripRecHitConverter") 
    << "found\n"	 
    << nmatch 
    << "  matched RecHits\n";
}

void SiStripRecHitConverterAlgorithm::
fillBad128StripBlocks(const uint32_t detid, bool bad128StripBlocks[6] ) const 
{
  if(maskBad128StripBlocks) {
    short badApvs   = quality->getBadApvs(detid);
    short badFibers = quality->getBadFibers(detid);
    for (int j = 0; j < 6; j++) {
      bad128StripBlocks[j] = (badApvs & (1 << j));
    }
    for (int j = 0; j < 3; j++) {
      if (badFibers & (1 << j)) {
	bad128StripBlocks[2*j+0] = true;
	bad128StripBlocks[2*j+1] = true;
      }
    }
  }
}

inline
bool SiStripRecHitConverterAlgorithm::
isMasked(const SiStripCluster &cluster, bool bad128StripBlocks[6]) const 
{
  if(maskBad128StripBlocks) {
    if ( bad128StripBlocks[cluster.firstStrip() >> 7] ) {
      if ( bad128StripBlocks[(cluster.firstStrip()+cluster.amplitudes().size())  >> 7] ||
	   bad128StripBlocks[static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7] ) {
	return true;
      }
    } else {
      if ( bad128StripBlocks[(cluster.firstStrip()+cluster.amplitudes().size())  >> 7] &&
	   bad128StripBlocks[static_cast<int32_t>(cluster.barycenter()-0.499999) >> 7] ) {
	return true;
      }
    }
  }
  return false;
}

inline
bool SiStripRecHitConverterAlgorithm::
useModule(const uint32_t id) const
{
  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker->idToDetUnit(id);
  if(stripdet==0) edm::LogWarning("SiStripRecHitConverter") << "Detid=" << id << " not found";
  return stripdet!=0 && (!useQuality || quality->IsModuleUsable(id));
}
