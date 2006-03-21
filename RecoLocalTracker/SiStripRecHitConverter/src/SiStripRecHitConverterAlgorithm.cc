// File: SiStripDetHitConverterAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  O/ Gutsche
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <algorithm>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"

#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

using namespace std;

SiStripRecHitConverterAlgorithm::SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
  clustermatch_=new SiStripRecHitMatcher();
}

SiStripRecHitConverterAlgorithm::~SiStripRecHitConverterAlgorithm() {
  if(clustermatch_!=0){
    delete clustermatch_;
  }
}
void SiStripRecHitConverterAlgorithm::run(const SiStripClusterCollection* input,SiStripRecHit2DMatchedLocalPosCollection & outmatched,SiStripRecHit2DLocalPosCollection & outrphi, SiStripRecHit2DLocalPosCollection & outstereo,const TrackerGeometry& tracker,const MagneticField &BField)
{
  run(input, outmatched,outrphi,outstereo,tracker,BField,LocalVector(0.,0.,0.));
}


void SiStripRecHitConverterAlgorithm::run(const SiStripClusterCollection* input,SiStripRecHit2DMatchedLocalPosCollection & outmatched,SiStripRecHit2DLocalPosCollection & outrphi, SiStripRecHit2DLocalPosCollection & outstereo,const TrackerGeometry& tracker,const MagneticField &BField,LocalVector trackdirection)
{
  const MagneticField *b=&BField;
  const TrackerGeometry *geom=&tracker;
  StripCPE parameterestimator(conf_,b,geom); 
  int nmono=0;
  int nstereo=0;
  int nmatch=0;
  int nunmatch=0;
  // get vector of detunit ids
  const std::vector<unsigned int> detIDs = input->detIDs();
  for ( std::vector<unsigned int>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); detunit_iterator++ ) {//loop over detectors
    //    bool isstereo=0;
    unsigned int id = *detunit_iterator;
    edm::OwnVector<SiStripRecHit2DLocalPos> collectorrphi; 
    edm::OwnVector<SiStripRecHit2DLocalPos> collectorstereo; 
    if(id!=999999999){ //if is valid detector
      DetId detId(id);
      //get geometry 
      const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
      if(stripdet==0)std::cout<<"Detid="<<id<<" not found, trying next one"<<endl;
      else{
	const SiStripClusterCollection::Range clusterRange = input->get(id);
	SiStripClusterCollection::ContainerIterator clusterRangeIteratorBegin = clusterRange.first;
	SiStripClusterCollection::ContainerIterator clusterRangeIteratorEnd   = clusterRange.second;
	SiStripClusterCollection::ContainerIterator iter;
	StripSubdetector specDetId=(StripSubdetector)(*detunit_iterator);
	for(iter=clusterRangeIteratorBegin;iter!=clusterRangeIteratorEnd;++iter){
	  //	float thickness=stripdet->specificSurface().bounds().thickness();
	  //GlobalVector gbfield=BField.inTesla(stripdet->surface().position());
	  //LocalVector drift=this->DriftDirection(stripdet,gbfield);
	  //drift*=thickness;
	  //std::cout<<"drift= "<<drift.mag()<<std::endl;
	  StripClusterParameterEstimator::LocalValues parameters=parameterestimator.localParameters(*iter);
	  std::vector<const SiStripCluster*> clusters;
	  clusters.push_back(&(*iter));
	  if(!specDetId.stereo()){
	    collectorrphi.push_back(new SiStripRecHit2DLocalPos(parameters.first, parameters.second,detId,clusters));
	    nmono++;
	  }
	  if(specDetId.stereo()){
	    collectorstereo.push_back(new SiStripRecHit2DLocalPos(parameters.first, parameters.second,detId,clusters));
	    nstereo++;
	  }
	}
	//      SiStripRecHit2DLocalPosCollection::range inputRangerphi(make_pair(collectorrphi.begin(),collectorrphi.end()));
	//      SiStripRecHit2DLocalPosCollection::range inputRangestereo(make_pair(collectorstereo.begin(),collectorstereo.end()));
	
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
  
  const std::vector<DetId> detIDs2 = outrphi.ids();
  for ( std::vector<DetId>::const_iterator detunit_iterator = detIDs2.begin(); detunit_iterator != detIDs2.end(); detunit_iterator++ ) {//loop over detectors
    edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collectorMatched; 
    SiStripRecHit2DLocalPosCollection::range monoRecHitRange = outrphi.get((*detunit_iterator));
    SiStripRecHit2DLocalPosCollection::const_iterator rhRangeIteratorBegin = monoRecHitRange.first;
    SiStripRecHit2DLocalPosCollection::const_iterator rhRangeIteratorEnd   = monoRecHitRange.second;
    SiStripRecHit2DLocalPosCollection::const_iterator iter;
    unsigned int id = 0;
    for(iter=rhRangeIteratorBegin;iter!=rhRangeIteratorEnd;++iter){//loop on the mono RH
      edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collectorMatchedSingleHit; 
      StripSubdetector specDetId(*detunit_iterator);
      id = specDetId.partnerDetId();
      std::vector<unsigned int>::const_iterator partnerdetiter=std::find(detIDs.begin(),detIDs.end(),id);
      if(partnerdetiter==detIDs.end()) id=0;	
      if (id>0){
	//	DetId partnerdetId(id);
	const GeomDetUnit * monostripdet=tracker.idToDetUnit(*detunit_iterator);
	const GeomDetUnit * stereostripdet=tracker.idToDetUnit(DetId(id));
	
	const SiStripRecHit2DLocalPosCollection::range rhpartnerRange = outstereo.get(DetId(id));
	SiStripRecHit2DLocalPosCollection::const_iterator rhpartnerRangeIteratorBegin = rhpartnerRange.first;
	SiStripRecHit2DLocalPosCollection::const_iterator rhpartnerRangeIteratorEnd   = rhpartnerRange.second;
	
	//	edm::OwnVector<SiStripRecHit2DMatchedLocalPos> tempCollector; 
	
	const DetId theId(id);
	const StripTopology& topol=(StripTopology&)stereostripdet->topology();
	collectorMatchedSingleHit=clustermatch_->match(&(*iter),rhpartnerRangeIteratorBegin,rhpartnerRangeIteratorEnd,theId,topol,monostripdet,stereostripdet,trackdirection);
	if (collectorMatchedSingleHit.size()>0) {
	  nmatch++;
	}else{
	  nunmatch++;
	  //std::cout<<"unmatched!"<<std::endl;
	}
	//	SiStripRecHit2DMatchedLocalPosCollection::Range inputRangematched(collectorMatched.begin(),collectorMatched.end());

	if (collectorMatchedSingleHit.size() > 0) {
	  for (    edm::OwnVector<SiStripRecHit2DMatchedLocalPos>::iterator itt = collectorMatchedSingleHit.begin();  itt != collectorMatchedSingleHit.end() ; itt++)
	    collectorMatched.push_back(new SiStripRecHit2DMatchedLocalPos(*itt));
	}
	//	for (edm::OwnVector<SiStripRecHit2DMatchedLocalPos>::iterator itt = tempCollector.begin(); itt!= tempCollector.end(); itt++)
	//	  collectorMatched.push_back(itt);

	//	copy(tempCollector.begin(), tempCollector.end(), back_inserter(collectorMatched.end()));
	
      
      }
    }
    if (collectorMatched.size()>0){
      //      outmatched.put(inputRangematched, *detunit_iterator);
      outmatched.put(DetId(*detunit_iterator),collectorMatched.begin(),collectorMatched.end());
    }
  }

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    std::cout << "[SiStripRecHitConverterAlgorithm] found" << std::endl; 
    std::cout << nmono << "  clusters in mono detectors"<< std::endl;
    std::cout << nstereo << "  clusters in partners stereo detectors"<< std::endl;
    std::cout << nmatch << "  matched RecHit"<< std::endl;
    std::cout << nunmatch << "  unmatched clusters "<< std::endl;
  }
};
  
