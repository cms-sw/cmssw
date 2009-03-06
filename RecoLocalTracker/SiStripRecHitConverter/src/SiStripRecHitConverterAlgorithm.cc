// File: SiStripRecHitConverterAlgorithm.cc
// Description:  Converts clusters into rechits
// Author:  C.Genta
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <algorithm>
#include <ext/algorithm>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"


//DataFormats
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/Ref.h"

//Geometry
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

//messagelogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

using namespace std;

SiStripRecHitConverterAlgorithm::SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

SiStripRecHitConverterAlgorithm::~SiStripRecHitConverterAlgorithm() {
}

void SiStripRecHitConverterAlgorithm::run(edm::Handle<edmNew::DetSetVector<SiStripCluster> >  input,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher, const SiStripQuality *quality)
{
  run(input, outmatched,outrphi,outstereo,tracker,parameterestimator,matcher,LocalVector(0.,0.,0.),quality);
}


void SiStripRecHitConverterAlgorithm::run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > inputhandle,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher,LocalVector trackdirection, const SiStripQuality *quality)
{

  int nmono=0;
  int nstereo=0;
  bool maskBad128StripBlocks, bad128StripBlocks[6];
  maskBad128StripBlocks = ((quality != 0) && conf_.existsAs<bool>("MaskBadAPVFibers") && conf_.getParameter<bool>("MaskBadAPVFibers"));

  std::vector<SiStripRecHit2D> collectorrphi, collectorstereo; 
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=inputhandle->begin(); DSViter!=inputhandle->end();DSViter++ ) {//loop over detectors
 
    unsigned int id = DSViter->id();
    collectorrphi.clear(); collectorstereo.clear();

    //    if(id!=999999999){ //if is valid detector
      DetId detId(id);
      //get geometry 
      const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
      if(stripdet==0)edm::LogWarning("SiStripRecHitConverter")<<"Detid="<<id<<" not found, trying next one";
      else if ((quality == 0) || quality->IsModuleUsable(detId)) { 
        if (maskBad128StripBlocks) fillBad128StripBlocks(*quality, detId, bad128StripBlocks);
        edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
        edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
        
        StripSubdetector specDetId=StripSubdetector(id);
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

          if(!specDetId.stereo()){ //if the cluster is in a mono det
            collectorrphi.push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
            nmono++;
          }
          else{                    //if the cluster in in stereo det
            collectorstereo.push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
            nstereo++;
          }
        }
        if (collectorrphi.size() > 0) {
          outrphi.put(detId,collectorrphi.begin(),collectorrphi.end());
        }
        if (collectorstereo.size() > 0) {
          outstereo.put(detId, collectorstereo.begin(),collectorstereo.end());
        }
      }
    }
  
  edm::LogInfo("SiStripRecHitConverter") 
    << "found\n"				 
    << nmono 			 
    << "  clusters in mono detectors\n"                            
    << nstereo  
    << "  clusters in partners stereo detectors\n";

  // Match the clusters
  match(outmatched,outrphi,outstereo,tracker,matcher,trackdirection);
}

void SiStripRecHitConverterAlgorithm::run(edm::Handle<edm::RefGetter<SiStripCluster> >  refGetterhandle, edm::Handle<edm::LazyGetter<SiStripCluster> >  lazyGetterhandle, SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher, const SiStripQuality *quality)
{
 
  int nmono=0;
  int nstereo=0;
  bool maskBad128StripBlocks, bad128StripBlocks[6];
  maskBad128StripBlocks = ((quality != 0) && conf_.existsAs<bool>("MaskBadAPVFibers") && conf_.getParameter<bool>("MaskBadAPVFibers"));

  edm::OwnVector<SiStripRecHit2D> collectorrphi; 
  edm::OwnVector<SiStripRecHit2D> collectorstereo;

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
	  collectorrphi.push_back(new SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	  nmono++;
	}
	else{           
	  collectorstereo.push_back(new SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	  nstereo++;
	}

	//If last cluster in det store clusters and clear OwnVector
	if (((icluster+1 )==iregion->end())
	    || ((icluster+1)->geographicalId() != icluster->geographicalId())) {

	  if (collectorrphi.size() > 0) {
	    outrphi.put(detId,collectorrphi.begin(),collectorrphi.end());
	    collectorrphi.clear();
	  }
	  
	  if (collectorstereo.size() > 0) {
	    outstereo.put(detId, collectorstereo.begin(),collectorstereo.end());
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
					

  match(outmatched,outrphi,outstereo,tracker,matcher,LocalVector(0.,0.,0.));
  
}


void SiStripRecHitConverterAlgorithm::match(SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker, const SiStripRecHitMatcher & matcher,LocalVector trackdirection) const {
  int maximumHits2BeforeMatching = conf_.getParameter<uint32_t>("maximumHits2BeforeMatching");
  bool skippedPairs = false;
  
  int nmatch=0;
  
  std::vector<DetId> rphidetIDs = outrphi.ids();
  std::vector<DetId> stereodetIDs = outstereo.ids();
  if (!__gnu_cxx::is_sorted(stereodetIDs.begin(), stereodetIDs.end())) {
        // this is an error in the logic of the RangeMap. Anyway, we can cope with it
        std::sort(stereodetIDs.begin(), stereodetIDs.end());
  }
  for ( std::vector<DetId>::const_iterator detunit_iterator = rphidetIDs.begin(); detunit_iterator != rphidetIDs.end(); detunit_iterator++ ) {//loop over detectors
    edm::OwnVector<SiStripMatchedRecHit2D> collectorMatched; 

    edm::OwnVector<SiStripMatchedRecHit2D> collectorMatchedSingleHit; 
    StripSubdetector specDetId(*detunit_iterator);
    unsigned int id = specDetId.partnerDetId();
    const DetId theId(id);
      
    //find if the detid of the stereo is in the list of stereo RH
    if (!std::binary_search(stereodetIDs.begin(),stereodetIDs.end(),theId)) id = 0;
    // Much better std::binary_search than std::find, as the list is sorted
    // was:// std::vector<DetId>::const_iterator partnerdetiter=std::binary_search(stereodetIDs.begin(),stereodetIDs.end(),theId);
    // was:// if(partnerdetiter==stereodetIDs.end()) id=0;	
 
    SiStripRecHit2DCollection::range monoRecHitRange = outrphi.get((*detunit_iterator));
    SiStripRecHit2DCollection::const_iterator rhRangeIteratorBegin = monoRecHitRange.first;
    SiStripRecHit2DCollection::const_iterator rhRangeIteratorEnd   = monoRecHitRange.second;
    SiStripRecHit2DCollection::const_iterator iter;
    
    for(iter=rhRangeIteratorBegin;iter!=rhRangeIteratorEnd;++iter){//loop over the mono RH
     
      if (id>0){ //if the detector has a stereo det associated and at least an hit in the stereo detector
	
	const SiStripRecHit2DCollection::range rhpartnerRange = outstereo.get(theId);
	SiStripRecHit2DCollection::const_iterator rhpartnerRangeIteratorBegin = rhpartnerRange.first;
	SiStripRecHit2DCollection::const_iterator rhpartnerRangeIteratorEnd   = rhpartnerRange.second;

        if ((maximumHits2BeforeMatching > 0) &&
            (monoRecHitRange.second - monoRecHitRange.first) * (rhpartnerRange.second - rhpartnerRange.first) > maximumHits2BeforeMatching) {
            skippedPairs = true;
            break;
        }
      
	const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(DetId(specDetId.glued()));
	SiStripRecHitMatcher::SimpleHitCollection stereoHits;
	stereoHits.reserve(rhpartnerRangeIteratorEnd-rhpartnerRangeIteratorBegin);

	for (SiStripRecHit2DCollection::const_iterator i=rhpartnerRangeIteratorBegin; i != rhpartnerRangeIteratorEnd; ++i) {
	  stereoHits.push_back( &(*i)); // convert to simple pointer
	}
	// perform the matchin looping over the hit on the stereo dets
	matcher.match(&(*iter),stereoHits.begin(),stereoHits.end(),collectorMatched,gluedDet,trackdirection);
	
      }
    }
    if (collectorMatched.size()>0){
      nmatch+=collectorMatched.size();
      StripSubdetector stripDetId(*detunit_iterator);
      outmatched.put(DetId(stripDetId.glued()),collectorMatched.begin(),collectorMatched.end());
    }
  }
  
  edm::LogInfo("SiStripRecHitConverter") 
    << "found\n"	 
    << nmatch 
    << "  matched RecHits\n";
  if (skippedPairs) {
    edm::LogWarning("SiStripRecHitConverter") << "Skipped matched rechits on some noisy modules.\n";
  }
}
void SiStripRecHitConverterAlgorithm::fillBad128StripBlocks(const SiStripQuality &quality, const uint32_t &detid, bool bad128StripBlocks[6]) const {
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
