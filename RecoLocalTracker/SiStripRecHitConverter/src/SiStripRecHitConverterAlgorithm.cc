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

using namespace std;

SiStripRecHitConverterAlgorithm::SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

SiStripRecHitConverterAlgorithm::~SiStripRecHitConverterAlgorithm() {
}

void SiStripRecHitConverterAlgorithm::run(edm::Handle<edmNew::DetSetVector<SiStripCluster> >  input,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher)
{
  run(input, outmatched,outrphi,outstereo,tracker,parameterestimator,matcher,LocalVector(0.,0.,0.));
}


void SiStripRecHitConverterAlgorithm::run(edm::Handle<edmNew::DetSetVector<SiStripCluster> > inputhandle,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher,LocalVector trackdirection)
{

  int nmono=0;
  int nstereo=0;

  std::vector<SiStripRecHit2D> collectorrphi, collectorstereo; 
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter=inputhandle->begin(); DSViter!=inputhandle->end();DSViter++ ) {//loop over detectors
 
    unsigned int id = DSViter->id();
    collectorrphi.clear(); collectorstereo.clear();

    //    if(id!=999999999){ //if is valid detector
      DetId detId(id);
      //get geometry 
      const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
      if(stripdet==0)edm::LogWarning("SiStripRecHitConverter")<<"Detid="<<id<<" not found, trying next one";
      else{
        edmNew::DetSet<SiStripCluster>::const_iterator begin=DSViter->begin();
        edmNew::DetSet<SiStripCluster>::const_iterator end  =DSViter->end();
        
        StripSubdetector specDetId=StripSubdetector(id);
        for(edmNew::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter){//loop over the clusters of the detector

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

void SiStripRecHitConverterAlgorithm::run(edm::Handle<edm::SiStripRefGetter<SiStripCluster> >  refGetterhandle, edm::Handle<edm::SiStripLazyGetter<SiStripCluster> >  lazyGetterhandle, SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher)
{
 
  int nmono=0;
  int nstereo=0;
  edm::OwnVector<SiStripRecHit2D> collectorrphi; 
  edm::OwnVector<SiStripRecHit2D> collectorstereo;
 
  edm::SiStripRefGetter<SiStripCluster>::const_iterator iregion = refGetterhandle->begin();
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
      else{
        
        StripSubdetector specDetId=StripSubdetector(icluster->geographicalId());
	StripClusterParameterEstimator::LocalValues parameters=parameterestimator.localParameters(*icluster,*stripdet);
	edm::Ref< edm::SiStripLazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> > cluster =
	  makeRefToSiStripLazyGetter(lazyGetterhandle,i);
       
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
}
