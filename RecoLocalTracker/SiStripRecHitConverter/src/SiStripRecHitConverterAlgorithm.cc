#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"

#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/Common/interface/Ref.h"

#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


void SiStripRecHitConverterAlgorithm::
run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > input,
    SiStripMatchedRecHit2DCollection & outmatched,
    SiStripRecHit2DCollection & outrphi, 
    SiStripRecHit2DCollection & outstereo, 
    SiStripRecHit2DCollection & outrphiUnmatched, 
    SiStripRecHit2DCollection & outstereoUnmatched,
    const TrackerGeometry& tracker,
    const StripClusterParameterEstimator &parameterestimator, 
    const SiStripRecHitMatcher & matcher, const SiStripQuality *quality)
{
  run(input, outmatched, outrphi, outstereo, outrphiUnmatched, outstereoUnmatched, tracker, parameterestimator, matcher, LocalVector(0.,0.,0.), quality);
}

void SiStripRecHitConverterAlgorithm::
run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > inputhandle,
    SiStripMatchedRecHit2DCollection & outmatched,
    SiStripRecHit2DCollection & outrphi, 
    SiStripRecHit2DCollection & outstereo, 
    SiStripRecHit2DCollection & outrphiUnmatched, 
    SiStripRecHit2DCollection & outstereoUnmatched,
    const TrackerGeometry& tracker,
    const StripClusterParameterEstimator &parameterestimator, 
    const SiStripRecHitMatcher & matcher,
    LocalVector trackdirection, const SiStripQuality *quality)
{
  int nmono=0;
  int nstereo=0;
  bool maskBad128StripBlocks, bad128StripBlocks[6];
  maskBad128StripBlocks = ((quality != 0) && conf_.existsAs<bool>("MaskBadAPVFibers") && conf_.getParameter<bool>("MaskBadAPVFibers"));

  for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=inputhandle->begin(); DSViter!=inputhandle->end();DSViter++ ) {//loop over detectors
 
    unsigned int id = DSViter->id();
    DetId detId(id);
    StripSubdetector specDetId=StripSubdetector(id);
    
    typedef SiStripRecHit2DCollection::FastFiller Collector;
    Collector collector = specDetId.stereo() ? Collector(outstereo, detId) : Collector(outrphi, detId);

      //get geometry 
      const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
      if(stripdet==0)edm::LogWarning("SiStripRecHitConverter")<<"Detid="<<id<<" not found, trying next one";
      else if ((quality == 0) || quality->IsModuleUsable(detId)) { 
        if (maskBad128StripBlocks) fillBad128StripBlocks(*quality, detId, bad128StripBlocks);
        edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
        edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
        
        for(edmNew::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter){//loop over the clusters of the detector
          // if masking is on, check that the cluster is not masked
          if (maskBad128StripBlocks && isMasked(*iter, bad128StripBlocks)) continue;

          //calculate the position and error in local coordinates
          StripClusterParameterEstimator::LocalValues parameters=parameterestimator.localParameters(*iter,*stripdet);

//           GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(parameters.first);
//           GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
//           LocalVector trackdir=(stripdet->surface()).toLocal(gtrackdirection);
//           const  LocalTrajectoryParameters trackparam=LocalTrajectoryParameters( parameters.first, trackdir,0);
//           parameters=parameterestimator.localParameters(*iter,*stripdet,trackparam);

          //store the ref to the cluster
          SiStripRecHit2D::ClusterRef cluster=edmNew::makeRefTo(inputhandle,iter);
          collector.push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
        }
      }
      if (specDetId.stereo()) nstereo += collector.size();
      else                    nmono   += collector.size();
      if (collector.empty()) collector.abort();
    }
  
  edm::LogInfo("SiStripRecHitConverter") 
    << "found\n"				 
    << nmono 			 
    << "  clusters in mono detectors\n"                            
    << nstereo  
    << "  clusters in partners stereo detectors\n";

  // Match the clusters
  match(outmatched,outrphi,outstereo,outrphiUnmatched,outstereoUnmatched,tracker,matcher,trackdirection);
}

void SiStripRecHitConverterAlgorithm::
run(edm::Handle<edm::RefGetter<SiStripCluster> >  refGetterhandle, 
    edm::Handle<edm::LazyGetter<SiStripCluster> >  lazyGetterhandle, 
    SiStripMatchedRecHit2DCollection & outmatched,
    SiStripRecHit2DCollection & outrphi, 
    SiStripRecHit2DCollection & outstereo, 
    SiStripRecHit2DCollection & outrphiUnmatched, 
    SiStripRecHit2DCollection & outstereoUnmatched,
    const TrackerGeometry& tracker,
    const StripClusterParameterEstimator &parameterestimator, 
    const SiStripRecHitMatcher & matcher, 
    const SiStripQuality *quality)
{
  int nmono=0;
  int nstereo=0;
  bool maskBad128StripBlocks, bad128StripBlocks[6];
  maskBad128StripBlocks = ((quality != 0) && conf_.existsAs<bool>("MaskBadAPVFibers") && conf_.getParameter<bool>("MaskBadAPVFibers"));

  std::vector<SiStripRecHit2D> collectorrphi; 
  std::vector<SiStripRecHit2D> collectorstereo;

  typedef SiStripRecHit2DCollection::FastFiller Collector;

  DetId lastId; bool goodDet = true;
 
  edm::RefGetter<SiStripCluster>::const_iterator iregion = refGetterhandle->begin();
  for(;iregion!=refGetterhandle->end();++iregion) {
    const edm::RegionIndex<SiStripCluster>& region = *iregion;
    const uint32_t start = region.start();
    const uint32_t finish = region.finish();
    for (uint32_t i = start; i < finish; i++) {
      edm::RegionIndex<SiStripCluster>::const_iterator icluster = region.begin()+(i-start);

      DetId detId(icluster->geographicalId());
      const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
      if(stripdet==0)
	edm::LogWarning("SiStripRecHitConverter")
	  <<"Detid="
	  <<icluster->geographicalId()
	  <<" not found";
       else {
        if (quality != 0) {
          if (detId != lastId) {
                lastId = detId;
                goodDet = quality->IsModuleUsable(detId);
                if (goodDet) fillBad128StripBlocks(*quality, detId, bad128StripBlocks);
          }
          if (!goodDet) continue;
          // if masking is on, check that the cluster is not masked
          if (maskBad128StripBlocks && isMasked(*icluster, bad128StripBlocks)) continue;
        } 

        StripSubdetector specDetId=StripSubdetector(icluster->geographicalId());
	StripClusterParameterEstimator::LocalValues parameters=parameterestimator.localParameters(*icluster,*stripdet);
	edm::Ref< edm::LazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> > cluster =
	  makeRefToLazyGetter(lazyGetterhandle,i);
       
	if(!specDetId.stereo()){ 
	  collectorrphi.push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	  nmono++;
	}
	else{           
	  collectorstereo.push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	  nstereo++;
	}

	//If last cluster in det store clusters and clear OwnVector
	if (((icluster+1 )==iregion->end())
	    || ((icluster+1)->geographicalId() != icluster->geographicalId())) {

	  if (collectorrphi.size() > 0) {
            Collector collector(outrphi, detId);
            collector.resize(collectorrphi.size());
            std::copy(collectorrphi.begin(),collectorrphi.end(),collector.begin());
	    collectorrphi.clear();
	  }
	  
	  if (collectorstereo.size() > 0) {
            Collector collector(outstereo, detId);
            collector.resize(collectorstereo.size());
            std::copy(collectorstereo.begin(),collectorstereo.end(),collector.begin());
	    collectorstereo.clear();
	  }	  
	}
      }
    }
  }
  
  edm::LogInfo("SiStripRecHitConverter") 
    << "found\n"				 
    << nmono 			 
    << "  clusters in mono detectors\n"                            
    << nstereo  
    << "  clusters in partners stereo detectors\n";
					

  match(outmatched,outrphi,outstereo,outrphiUnmatched,outstereoUnmatched,tracker,matcher,LocalVector(0.,0.,0.));
  
}



void SiStripRecHitConverterAlgorithm::
match(SiStripMatchedRecHit2DCollection & outmatched,
      SiStripRecHit2DCollection & outrphi, 
      SiStripRecHit2DCollection & outstereo, 
      SiStripRecHit2DCollection & outrphiUnmatched, 
      SiStripRecHit2DCollection & outstereoUnmatched,
      const TrackerGeometry& tracker, 
      const SiStripRecHitMatcher & matcher,
      LocalVector trackdirection) const 
{
  int nmatch=0;
  edm::OwnVector<SiStripMatchedRecHit2D> collectorMatched; // gp/FIXME: avoid this

  // Remember the ends of the collections, as we will use them a lot
  SiStripRecHit2DCollection::const_iterator edStereoDet = outstereo.end();
  SiStripRecHit2DCollection::const_iterator edRPhiDet   = outrphi.end();

  // two work vectors for bookeeping clusters used by the stereo part of the matched hits
  std::vector<SiStripRecHit2D::ClusterRef::key_type>         matchedSteroClusters;
  std::vector<SiStripRecHit2D::ClusterRegionalRef::key_type> matchedSteroClustersRegional;

  for (SiStripRecHit2DCollection::const_iterator itRPhiDet = outrphi.begin(); itRPhiDet != edRPhiDet; ++itRPhiDet) {
    edmNew::DetSet<SiStripRecHit2D> rphiHits = *itRPhiDet;
    StripSubdetector specDetId(rphiHits.detId());
    uint32_t partnerId = specDetId.partnerDetId();

    // if not part of a glued pair
    if (partnerId == 0) { 
        // I must copy these as unmatched 
        if (!rphiHits.empty()) {
            SiStripRecHit2DCollection::FastFiller filler(outrphiUnmatched, rphiHits.detId());
            filler.resize(rphiHits.size());
            std::copy(rphiHits.begin(), rphiHits.end(), filler.begin());
        }
        continue;
    }

    SiStripRecHit2DCollection::const_iterator itStereoDet = outstereo.find(partnerId);

    // if the partner is not found (which probably can happen if it's empty)
    if (itStereoDet == edStereoDet) {
        // I must copy these as unmatched 
        if (!rphiHits.empty()) {
            SiStripRecHit2DCollection::FastFiller filler(outrphiUnmatched, rphiHits.detId());
            filler.resize(rphiHits.size());
            std::copy(rphiHits.begin(), rphiHits.end(), filler.begin());
        }
        continue;
    }

    edmNew::DetSet<SiStripRecHit2D> stereoHits = *itStereoDet;

    // Make simple collection of this (gp:FIXME: why do we need it?)
    SiStripRecHitMatcher::SimpleHitCollection stereoSimpleHits;
    // gp:FIXME: use std::transform 
    stereoSimpleHits.reserve(stereoHits.size());
    for (edmNew::DetSet<SiStripRecHit2D>::const_iterator it = stereoHits.begin(), ed = stereoHits.end(); it != ed; ++it) {
        stereoSimpleHits.push_back(&*it);
    }

    // Get ready for making glued hits
    const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(DetId(specDetId.glued()));
    typedef SiStripMatchedRecHit2DCollection::FastFiller Collector;
    Collector collector(outmatched, specDetId.glued());

    // Prepare also the list for unmatched rphi hits
    SiStripRecHit2DCollection::FastFiller fillerRphiUnm(outrphiUnmatched, rphiHits.detId());

    // a list of clusters used by the matched part of the stereo hits in this detector
    matchedSteroClusters.clear();          // at the beginning, empty
    matchedSteroClustersRegional.clear();  // I need two because the refs can be different
    bool regional = false;                 // I also want to remember if they come from standard or HLT reco

    for (edmNew::DetSet<SiStripRecHit2D>::const_iterator it = rphiHits.begin(), ed = rphiHits.end(); it != ed; ++it) {
	matcher.match(&(*it),stereoSimpleHits.begin(),stereoSimpleHits.end(),collectorMatched,gluedDet,trackdirection);
        if (collectorMatched.size()>0){
          nmatch+=collectorMatched.size();
          for (edm::OwnVector<SiStripMatchedRecHit2D>::const_iterator itm = collectorMatched.begin(),
                                                                      edm = collectorMatched.end();
                itm != edm; 
                ++itm) {
            collector.push_back(*itm);
            // mark the stereo hit cluster as used, so that the hit won't go in the unmatched stereo ones
            if (itm->stereoHit()->cluster().isNonnull()) {
                matchedSteroClusters.push_back(itm->stereoHit()->cluster().key()); 
            } else {
                matchedSteroClustersRegional.push_back(itm->stereoHit()->cluster_regional().key()); 
                regional = true;
            }
          }
          collectorMatched.clear();
        } else {
          // store a copy of this rphi hit as an unmatched rphi hit
          fillerRphiUnm.push_back(*it);
        }
    }

    // discard matched hits if the collection is empty
    if (collector.empty()) collector.abort();

    // discard unmatched rphi hits if there are none
    if (fillerRphiUnm.empty()) fillerRphiUnm.abort();

    // now look for unmatched stereo hits    
    SiStripRecHit2DCollection::FastFiller fillerStereoUnm(outstereoUnmatched, stereoHits.detId());
    if (!regional) {
        std::sort(matchedSteroClusters.begin(), matchedSteroClusters.end());
        for (edmNew::DetSet<SiStripRecHit2D>::const_iterator it = stereoHits.begin(), ed = stereoHits.end(); it != ed; ++it) {
            if (!std::binary_search(matchedSteroClusters.begin(), matchedSteroClusters.end(), it->cluster().key())) {
                fillerStereoUnm.push_back(*it);
            }
        }
    } else {
        std::sort(matchedSteroClustersRegional.begin(), matchedSteroClustersRegional.end());
        for (edmNew::DetSet<SiStripRecHit2D>::const_iterator it = stereoHits.begin(), ed = stereoHits.end(); it != ed; ++it) {
            if (!std::binary_search(matchedSteroClustersRegional.begin(), matchedSteroClustersRegional.end(), it->cluster_regional().key())) {
                fillerStereoUnm.push_back(*it);
            }
        }
    }
    if (fillerStereoUnm.empty()) fillerStereoUnm.abort(); 
    

  }
  
  for (SiStripRecHit2DCollection::const_iterator itStereoDet = outstereo.begin(); itStereoDet != edStereoDet; ++itStereoDet) {
      edmNew::DetSet<SiStripRecHit2D> stereoHits = *itStereoDet;
      StripSubdetector specDetId(stereoHits.detId());
      uint32_t partnerId = specDetId.partnerDetId();
      if (partnerId == 0) continue;
      SiStripRecHit2DCollection::const_iterator itRPhiDet = outrphi.find(partnerId);
      if (itRPhiDet == edRPhiDet) {
          if (!stereoHits.empty()) {
              SiStripRecHit2DCollection::FastFiller filler(outstereoUnmatched, stereoHits.detId());
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
fillBad128StripBlocks(const SiStripQuality &quality, 
		      const uint32_t &detid, 
		      bool bad128StripBlocks[6] ) const 
{
    short badApvs   = quality.getBadApvs(detid);
    short badFibers = quality.getBadFibers(detid);
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


inline
bool SiStripRecHitConverterAlgorithm::
isMasked(const SiStripCluster &cluster, 
	 bool bad128StripBlocks[6]) const 
{
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
  return false;
}
