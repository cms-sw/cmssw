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


void SiStripRecHitConverterAlgorithm::run(edm::Handle<edm::DetSetVector<SiStripCluster> >  inputhandle,SiStripMatchedRecHit2DCollection & outmatched,SiStripRecHit2DCollection & outrphi, SiStripRecHit2DCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher,LocalVector trackdirection)
{

  int nmono=0;
  int nstereo=0;
  int nmatch=0;
  int nunmatch=0;

  const edm::DetSetVector<SiStripCluster>& input = *inputhandle;

  for (edm::DetSetVector<SiStripCluster>::const_iterator DSViter=input.begin(); DSViter!=input.end();DSViter++ ) {//loop over detectors

    unsigned int id = DSViter->id;

    edm::OwnVector<SiStripRecHit2D> collectorrphi; 
    edm::OwnVector<SiStripRecHit2D> collectorstereo; 
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

          GlobalPoint gcenterofstrip=(stripdet->surface()).toGlobal(parameters.first);
          GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
          LocalVector trackdir=(stripdet->surface()).toLocal(gtrackdirection);
          const  LocalTrajectoryParameters trackparam=LocalTrajectoryParameters( parameters.first, trackdir,0);
          parameters=parameterestimator.localParameters(*iter,*stripdet,trackparam);

	  //store the ref to the cluster
	  edm::Ref< edm::DetSetVector <SiStripCluster>,SiStripCluster > const & cluster=edm::makeRefTo(inputhandle,id,iter);

	  if(!specDetId.stereo()){ //if the cluster is in a mono det
	    collectorrphi.push_back(new SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
	    nmono++;
	  }
	  else{                    //if the cluster in in stereo det
	    collectorstereo.push_back(new SiStripRecHit2D(parameters.first, parameters.second,detId,cluster));
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
  //  }
  //
  // match the clusters
  //
  
  const std::vector<DetId> rphidetIDs = outrphi.ids();
  const std::vector<DetId> stereodetIDs = outstereo.ids();
  for ( std::vector<DetId>::const_iterator detunit_iterator = rphidetIDs.begin(); detunit_iterator != rphidetIDs.end(); detunit_iterator++ ) {//loop over detectors
    edm::OwnVector<SiStripMatchedRecHit2D> collectorMatched; 
    SiStripRecHit2DCollection::range monoRecHitRange = outrphi.get((*detunit_iterator));
    SiStripRecHit2DCollection::const_iterator rhRangeIteratorBegin = monoRecHitRange.first;
    SiStripRecHit2DCollection::const_iterator rhRangeIteratorEnd   = monoRecHitRange.second;
    SiStripRecHit2DCollection::const_iterator iter;

    unsigned int id = 0;
    for(iter=rhRangeIteratorBegin;iter!=rhRangeIteratorEnd;++iter){//loop over the mono RH
      edm::OwnVector<SiStripMatchedRecHit2D> collectorMatchedSingleHit; 
      StripSubdetector specDetId(*detunit_iterator);
      id = specDetId.partnerDetId();
      const DetId theId(id);

      //find if the detid of the stereo is in the list of stereo RH
      std::vector<DetId>::const_iterator partnerdetiter=std::find(stereodetIDs.begin(),stereodetIDs.end(),theId);
      if(partnerdetiter==stereodetIDs.end()) id=0;	

      const SiStripRecHit2DCollection::range rhpartnerRange = outstereo.get(theId);
      SiStripRecHit2DCollection::const_iterator rhpartnerRangeIteratorBegin = rhpartnerRange.first;
      SiStripRecHit2DCollection::const_iterator rhpartnerRangeIteratorEnd   = rhpartnerRange.second;
      
      
      if (id>0){ //if the detector has a stereo det associated and at least an  hit in the stereo detector

	const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(DetId(specDetId.glued()));

	// perform the matchin looping over the hit on the stereo dets
	collectorMatchedSingleHit=matcher.match(&(*iter),rhpartnerRangeIteratorBegin,rhpartnerRangeIteratorEnd,gluedDet,trackdirection);

	if (collectorMatchedSingleHit.size() > 0) { //if a matched is found add the hit to the temporary collection
	  nmatch++;
	  for (    edm::OwnVector<SiStripMatchedRecHit2D>::iterator itt = collectorMatchedSingleHit.begin();  itt != collectorMatchedSingleHit.end() ; itt++)
	    collectorMatched.push_back(new SiStripMatchedRecHit2D(*itt));
	}
	else{
	  nunmatch++;
	}
      }
    }
  if (collectorMatched.size()>0){
    StripSubdetector stripDetId(*detunit_iterator);
    outmatched.put(DetId(stripDetId.glued()),collectorMatched.begin(),collectorMatched.end());
  }
  }
  
  edm::LogInfo("SiStripRecHitConverter") << "found\n"
					 << nmono << "  clusters in mono detectors\n"
					 << nstereo << "  clusters in partners stereo detectors\n"
					 << nmatch << "  matched RecHit\n"
					 << nunmatch << "  unmatched clusters";
}

