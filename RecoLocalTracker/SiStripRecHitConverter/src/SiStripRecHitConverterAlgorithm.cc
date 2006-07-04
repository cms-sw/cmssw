// File: SiStripDetHitConverterAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  O/ Gutsche
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

void SiStripRecHitConverterAlgorithm::run(edm::Handle<edm::DetSetVector<SiStripCluster> >  input,SiStripRecHit2DMatchedLocalPosCollection & outmatched,SiStripRecHit2DLocalPosCollection & outrphi, SiStripRecHit2DLocalPosCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher)
{
  run(input, outmatched,outrphi,outstereo,tracker,parameterestimator,matcher,LocalVector(0.,0.,0.));
}


void SiStripRecHitConverterAlgorithm::run(edm::Handle<edm::DetSetVector<SiStripCluster> >  inputhandle,SiStripRecHit2DMatchedLocalPosCollection & outmatched,SiStripRecHit2DLocalPosCollection & outrphi, SiStripRecHit2DLocalPosCollection & outstereo,const TrackerGeometry& tracker,const StripClusterParameterEstimator &parameterestimator, const SiStripRecHitMatcher & matcher,LocalVector trackdirection)
{
  std::cout<<"Produce new event..."<<std::endl;

  int nmono=0;
  int nstereo=0;
  int nmatch=0;
  int nunmatch=0;

  const edm::DetSetVector<SiStripCluster>& input = *inputhandle;

  for (edm::DetSetVector<SiStripCluster>::const_iterator DSViter=input.begin(); DSViter!=input.end();DSViter++ ) {//loop over detectors

    unsigned int id = DSViter->id;

    edm::OwnVector<SiStripRecHit2DLocalPos> collectorrphi; 
    edm::OwnVector<SiStripRecHit2DLocalPos> collectorstereo; 
    if(id!=999999999){ //if is valid detector
      DetId detId(id);
      //get geometry 
      const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
      if(stripdet==0)edm::LogWarning("SiStripRecHitConverter")<<"Detid="<<id<<" not found, trying next one";
      else{
	edm::DetSet<SiStripCluster>::const_iterator begin=DSViter->data.begin();
	edm::DetSet<SiStripCluster>::const_iterator end  =DSViter->data.end();
	
	StripSubdetector specDetId=StripSubdetector(id);
	for(edm::DetSet<SiStripCluster>::const_iterator iter=begin;iter!=end;++iter){
	  StripClusterParameterEstimator::LocalValues parameters=parameterestimator.localParameters(*iter,*stripdet);

	  edm::Ref< edm::DetSetVector <SiStripCluster>,SiStripCluster > const & cluster=edm::makeRefTo(inputhandle,id,iter);

	  if(!specDetId.stereo()){
	    collectorrphi.push_back(new SiStripRecHit2DLocalPos(parameters.first, parameters.second,detId,cluster));
	    nmono++;
	  }
	  if(specDetId.stereo()){
	    collectorstereo.push_back(new SiStripRecHit2DLocalPos(parameters.first, parameters.second,detId,cluster));
	    nstereo++;
	  }
	}
	if (collectorrphi.size() > 0) {
	  outrphi.put(DetId(id),collectorrphi.begin(),collectorrphi.end());
	}
	if (collectorstereo.size() > 0) {
	  outstereo.put(DetId(id), collectorstereo.begin(),collectorstereo.end());
	}
      }
    }
  }
  //
  // match the clusters
  //
  
  const std::vector<DetId> rphidetIDs = outrphi.ids();
  const std::vector<DetId> stereodetIDs = outstereo.ids();
  for ( std::vector<DetId>::const_iterator detunit_iterator = rphidetIDs.begin(); detunit_iterator != rphidetIDs.end(); detunit_iterator++ ) {//loop over detectors
    edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collectorMatched; 
    SiStripRecHit2DLocalPosCollection::range monoRecHitRange = outrphi.get((*detunit_iterator));
    SiStripRecHit2DLocalPosCollection::const_iterator rhRangeIteratorBegin = monoRecHitRange.first;
    SiStripRecHit2DLocalPosCollection::const_iterator rhRangeIteratorEnd   = monoRecHitRange.second;
    SiStripRecHit2DLocalPosCollection::const_iterator iter;
    //int numrechitrphi = rhRangeIteratorEnd - rhRangeIteratorBegin;
    //cout<<"n rechit= "<<numrechitrphi<<endl;
    unsigned int id = 0;
    for(iter=rhRangeIteratorBegin;iter!=rhRangeIteratorEnd;++iter){//loop on the mono RH
      edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collectorMatchedSingleHit; 
      StripSubdetector specDetId(*detunit_iterator);
      id = specDetId.partnerDetId();
      const DetId theId(id);
      std::vector<DetId>::const_iterator partnerdetiter=std::find(stereodetIDs.begin(),stereodetIDs.end(),theId);
      if(partnerdetiter==stereodetIDs.end()) id=0;	
      const SiStripRecHit2DLocalPosCollection::range rhpartnerRange = outstereo.get(DetId(id));
      SiStripRecHit2DLocalPosCollection::const_iterator rhpartnerRangeIteratorBegin = rhpartnerRange.first;
      SiStripRecHit2DLocalPosCollection::const_iterator rhpartnerRangeIteratorEnd   = rhpartnerRange.second;
      
      
      if (id>0){

	const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(DetId(specDetId.glued()));
	std::cout<<"Perform the matching..."<<std::endl;
	collectorMatchedSingleHit=matcher.match(&(*iter),rhpartnerRangeIteratorBegin,rhpartnerRangeIteratorEnd,gluedDet,trackdirection);
	if (collectorMatchedSingleHit.size()>0) {
	  nmatch++;
	}else{
	  nunmatch++;
      }
	
	if (collectorMatchedSingleHit.size() > 0) {
	  for (    edm::OwnVector<SiStripRecHit2DMatchedLocalPos>::iterator itt = collectorMatchedSingleHit.begin();  itt != collectorMatchedSingleHit.end() ; itt++)
	    collectorMatched.push_back(new SiStripRecHit2DMatchedLocalPos(*itt));
	}
      }
    }
  if (collectorMatched.size()>0){

    outmatched.put(DetId(*detunit_iterator),collectorMatched.begin(),collectorMatched.end());
  }
  }
  
  edm::LogInfo("SiStripRecHitConverter") << "found\n"
					 << nmono << "  clusters in mono detectors\n"
					 << nstereo << "  clusters in partners stereo detectors\n"
					 << nmatch << "  matched RecHit\n"
					 << nunmatch << "  unmatched clusters";
};

