// File: SiStripRecHitConverterAlgorithm.cc
// Description:  Converts clusters into rechits
// Author:  C.Genta
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <algorithm>
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

void SiStripRecHitConverterAlgorithm::run(edm::Handle<edm::DetSetVector<SiStripCluster> >  input,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher)
{
  run(input, outmatched,outrphi,outstereo,tracker,parameterestimator,matcher,LocalVector(0.,0.,0.));
}


void SiStripRecHitConverterAlgorithm::run(edm::Handle<edm::DetSetVector<SiStripCluster> > inputhandle,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher,LocalVector trackdirection)
{
  
  int nmono=0;
  int nstereo=0;
  
  for (edm::DetSetVector<SiStripCluster>::const_iterator DSViter=inputhandle->begin(); DSViter!=inputhandle->end();DSViter++ ) {//loop over detectors
    
    unsigned int id = DSViter->id;
    
    SiStripRecHit2DCollection::FastFiller collectorrphi(outrphi,id); 
    SiStripRecHit2DCollection::FastFiller collectorstereo(outstereo,id);
    //    if(id!=999999999){ //if is valid detector
    DetId detId(id);
    //get geometry 
    const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
    if(stripdet==0)edm::LogWarning("SiStripRecHitConverter")<<"Detid="<<id<<" not found, trying next one";
    else{
      edm::DetSet<SiStripCluster>::const_iterator begin=DSViter->data.begin();
      edm::DetSet<SiStripCluster>::const_iterator end  =DSViter->data.end();
      
      StripSubdetector specDetId=StripSubdetector(id);
      for(edm::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter){//loop over the clusters of the detector
	
	//calculate the position and error in local coordinates
	StripClusterParameterEstimator::LocalValues parameters=parameterestimator.localParameters(*iter,*stripdet);
	
	//           GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(parameters.first);
	//           GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
	//           LocalVector trackdir=(stripdet->surface()).toLocal(gtrackdirection);
	//           const  LocalTrajectoryParameters trackparam=LocalTrajectoryParameters( parameters.first, trackdir,0);
	//           parameters=parameterestimator.localParameters(*iter,*stripdet,trackparam);
	
	//store the ref to the cluster
	edm::Ref< edm::DetSetVector <SiStripCluster>,SiStripCluster > const & cluster=edm::makeRefTo(inputhandle,id,iter);
	
	if(!specDetId.stereo()){ //if the cluster is in a mono det
	  collectorrphi.push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	  nmono++;
	}
	else{                    //if the cluster in in stereo det
	  collectorstereo.push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	  nstereo++;
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
  
  // Match the clusters
  match(outmatched,outrphi,outstereo,tracker,matcher,trackdirection);
}

void SiStripRecHitConverterAlgorithm::run(edm::Handle<edm::SiStripRefGetter<SiStripCluster> >  inputhandle,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher)
{
 
  int nmono=0;
  int nstereo=0;
  
  
  edm::SiStripRefGetter<SiStripCluster>::const_iterator iregion = inputhandle->begin();
  for(;iregion!=inputhandle->end();++iregion) {
    vector<SiStripCluster>::const_iterator icluster = iregion->begin();
    
    DetId detIdold(0);
    typedef SiStripRecHit2DCollection::FastFiller Coll;
    typedef std::auto_ptr<Coll>  CollPointer;
    CollPointer collectorrphi;
    CollPointer collectorstereo;
    for (;icluster!=iregion->end();icluster++) {
      DetId detId(icluster->geographicalId());
      if (detId!=detIdold) {
	detIdold=detId;
	collectorrphi.reset(new Coll(outrphi,detId)); 
	collectorstereo.reset(new Coll(outstereo,detId));
      }
      
      const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
      if(stripdet==0)
	edm::LogWarning("SiStripRecHitConverter")
	  <<"Detid="
	  <<icluster->geographicalId()
	  <<" not found";
      else{
        
        StripSubdetector specDetId=StripSubdetector(icluster->geographicalId());
	StripClusterParameterEstimator::LocalValues parameters=parameterestimator.localParameters(*icluster,*stripdet);
	edm::SiStripRefGetter<SiStripCluster>::value_ref cluster = edm::makeRefToSiStripRefGetter(inputhandle,icluster);
	
	if(!specDetId.stereo()){ 
	  collectorrphi->push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	  nmono++;
	}
	else{           
	  collectorstereo->push_back(SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	  nstereo++;
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


void SiStripRecHitConverterAlgorithm::match(SiStripMatchedRecHit2DCollection & outmatched, SiStripRecHit2DCollection & outrphi, 
					    SiStripRecHit2DCollection & outstereo, const TrackerGeometry& tracker, 
					    const SiStripRecHitMatcher & matcher, LocalVector trackdirection) const {
  
  int nmatch=0;
  int nunmatch=0;
  
  for ( SiStripRecHit2DCollection::const_iterator pdetset = outrphi.begin(); pdetset != outrphi.end(); ++pdetset ) {//loop over detectors
    SiStripRecHit2DCollection::DetSet detset = *pdetset;
    
    StripSubdetector specDetId(detset.id());
    unsigned int id = specDetId.partnerDetId();
    SiStripMatchedRecHit2DCollection::FastFiller collectorMatched(outmatched, specDetId.glued());
    
    //find if the detid of the stereo is in the list of stereo RH
    SiStripRecHit2DCollection::const_iterator partnerIter = outstereo.find(id);
    if(partnerIter==outstereo.end()) continue;	
    
    for (SiStripRecHit2DCollection::DetSet::const_iterator iter = detset.begin(); iter!=detset.end(); ++iter) {
      SiStripRecHit2DCollection::DetSet partnerDetset = *partnerIter;
      
      const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(specDetId.glued());
      

      size_t cs = collectorMatched.size();
      // perform the matchin looping over the hit on the stereo det
      matcher.match(&(*iter),partnerDetset.begin(),partnerDetset.end(),
		    collectorMatched, gluedDet,trackdirection);

      
      if (collectorMatched.size() > cs) { //if a matched is found add the hit to the temporary collection
	nmatch++;
      }
      else{
	nunmatch++;
      }
    }
  }
  
  
  edm::LogInfo("SiStripRecHitConverter") 
    << "found\n"	 
    << nmatch 
    << "  matched RecHit\n"
    << nunmatch 
    << "  unmatched clusters";
}
