// File: SiStripDetHitConverterAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  O/ Gutsche
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <algorithm>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"

#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

using namespace std;

SiStripRecHitConverterAlgorithm::SiStripRecHitConverterAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
  clustermatch_=new SiStripClusterMatch();
}

SiStripRecHitConverterAlgorithm::~SiStripRecHitConverterAlgorithm() {
  if(clustermatch_!=0){
    delete clustermatch_;
  }
}


void SiStripRecHitConverterAlgorithm::run(const SiStripClusterCollection* input,SiStripRecHit2DLocalPosCollection & outmatched,SiStripRecHit2DLocalPosCollection & outrphi, SiStripRecHit2DLocalPosCollection & outstereo,const TrackingGeometry& tracker)
{
  int nmono=0;
  int nstereo=0;
  int nmatch=0;
  int nunmatch=0;
  // get vector of detunit ids
  const std::vector<unsigned int> detIDs = input->detIDs();
  for ( std::vector<unsigned int>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); detunit_iterator++ ) {//loop over detectors
    bool isstereo=0;
    unsigned int id = *detunit_iterator;
    own_vector<SiStripRecHit2DLocalPos> collectormatched; 
    own_vector<SiStripRecHit2DLocalPos> collectorrphi; 
    own_vector<SiStripRecHit2DLocalPos> collectorstereo; 
    if(id!=999999999){ //if is valid detector
      DetId detId(id);
      //get geometry 
      const GeomDetUnit * stripdet=tracker.idToDet(detId);
      const SiStripClusterCollection::Range clusterRange = input->get(id);
      SiStripClusterCollection::ContainerIterator clusterRangeIteratorBegin = clusterRange.first;
      SiStripClusterCollection::ContainerIterator clusterRangeIteratorEnd   = clusterRange.second;
      SiStripClusterCollection::ContainerIterator iter;
      if(detId.subdetId()==StripSubdetector::TIB||detId.subdetId()==StripSubdetector::TOB){// if TIB of TOB use the rectangualr strip topology 
	const RectangularStripTopology& Rtopol=(RectangularStripTopology&)stripdet->topology();
	// get the partner id only if the detector is r-phi
	unsigned int partnerid=0;
	TIBDetId TIBspecDetId(detId.rawId());
	TOBDetId TOBspecDetId(detId.rawId());
	switch(detId.subdetId()){
	case StripSubdetector::TIB:
	  if(!TIBspecDetId.stereo())partnerid=TIBspecDetId.partnerDetId();
	  else isstereo=1;
	  break;
	case StripSubdetector::TOB:
	  if(!TOBspecDetId.stereo())partnerid=TOBspecDetId.partnerDetId();
	  else isstereo=1;
	  break;
	}
	std::vector<unsigned int>::const_iterator partnerdetiter=std::find(detIDs.begin(),detIDs.end(),partnerid);
	if(partnerdetiter==detIDs.end()) partnerid=0;	
	for(iter=clusterRangeIteratorBegin;iter!=clusterRangeIteratorEnd;++iter){//loop on the cluster
	  SiStripCluster cluster=*iter;
	  LocalPoint position=Rtopol.localPosition(cluster.barycenter());
	  const  LocalError dummy;
	  std::vector<const SiStripCluster*> clusters;
	  clusters.push_back(&cluster);
	  if(!isstereo){
	    collectorrphi.push_back(new SiStripRecHit2DLocalPos(position, dummy, stripdet,detId,clusters));
	    nmono++;
	  }
	  if(isstereo){
	    collectorstereo.push_back(new SiStripRecHit2DLocalPos(position, dummy, stripdet,detId,clusters));
	    nstereo++;
	  }
	  if(partnerid!=0){ //If exist a cluster in the partner det
	    bool ismatch=false;
	    DetId partnerdetId(partnerid);
	    const GeomDetUnit * partnerstripdet=tracker.idToDet(partnerdetId);
	    const SiStripClusterCollection::Range clusterpartnerRange = input->get(partnerid);
	    SiStripClusterCollection::ContainerIterator clusterpartnerRangeIteratorBegin = clusterpartnerRange.first;
	    SiStripClusterCollection::ContainerIterator clusterpartnerRangeIteratorEnd   = clusterpartnerRange.second;
	    collectormatched=clustermatch_->match<RectangularStripTopology>(&cluster,clusterpartnerRangeIteratorBegin,clusterpartnerRangeIteratorEnd,detId,Rtopol,stripdet,partnerstripdet);
	    if(collectormatched.size()>0)nmatch+=collectormatched.size();
	    else nunmatch++;
	  }
	}
      }
      else if (detId.subdetId()==(StripSubdetector::TID)||detId.subdetId()==(StripSubdetector::TEC)){    //if TID or TEC use trapoeziodalstrip topology
	const TrapezoidalStripTopology& Ttopol=(TrapezoidalStripTopology&)stripdet->topology();
	// get the partner id only if the detector is r-phi
	unsigned int partnerid=0;
	TIDDetId TIDspecDetId(detId.rawId());
	TECDetId TECspecDetId(detId.rawId());
	switch(detId.subdetId()){
	case StripSubdetector::TID:
	  if(!TIDspecDetId.stereo())partnerid=TIDspecDetId.partnerDetId();
	  break;
	case StripSubdetector::TEC:
	  if(!TECspecDetId.stereo())partnerid=TECspecDetId.partnerDetId();
	  break;
	}
	std::vector<unsigned int>::const_iterator partnerdetiter=std::find(detIDs.begin(),detIDs.end(),partnerid);
	if(partnerdetiter==detIDs.end()) partnerid=0;	
	for(iter=clusterRangeIteratorBegin;iter!=clusterRangeIteratorEnd;++iter){//loop on the cluster
	  SiStripCluster cluster=*iter;
	  LocalPoint position=Ttopol.localPosition(cluster.barycenter());
	  const  LocalError dummy;
	  std::vector<const SiStripCluster*> clusters;
	  clusters.push_back(&cluster);
	  if(!isstereo){
	    collectorrphi.push_back(new SiStripRecHit2DLocalPos(position, dummy, stripdet,detId,clusters));
	    nmono++;
	  }
	  if(isstereo){
	    collectorstereo.push_back(new SiStripRecHit2DLocalPos(position, dummy, stripdet,detId,clusters));
	    nstereo++;
	  }
	  if(partnerid!=0){ //If exist a cluster in the partner det
	    bool ismatch=false;
	    DetId partnerdetId(partnerid);
	    const GeomDetUnit * partnerstripdet=tracker.idToDet(partnerdetId);
	    const SiStripClusterCollection::Range clusterpartnerRange = input->get(partnerid);
	    SiStripClusterCollection::ContainerIterator clusterpartnerRangeIteratorBegin = clusterpartnerRange.first;
	    SiStripClusterCollection::ContainerIterator clusterpartnerRangeIteratorEnd   = clusterpartnerRange.second;
	    collectormatched=clustermatch_->match<TrapezoidalStripTopology>(&cluster,clusterpartnerRangeIteratorBegin,clusterpartnerRangeIteratorEnd,detId,Ttopol,stripdet,partnerstripdet);
	    if(collectormatched.size()>0)nmatch+=collectormatched.size();
	    else nunmatch++;
	  }
	}

      }
      SiStripRecHit2DLocalPosCollection::Range inputRangematched(collectormatched.begin(),collectormatched.end());
      SiStripRecHit2DLocalPosCollection::Range inputRangerphi(collectorrphi.begin(),collectorrphi.end());
      SiStripRecHit2DLocalPosCollection::Range inputRangestereo(collectorstereo.begin(),collectorstereo.end());
      //      inputRange.first = collectormatched.begin();
      //inputRange.second = collectormatched.end();
      //outmatched.put(inputRange,id);
      if (collectormatched.size() > 0) {
        outmatched.put(inputRangematched,id);
      }
      if (collectorrphi.size() > 0) {
        outrphi.put(inputRangerphi,id);
      }
      if (collectorstereo.size() > 0) {
        outstereo.put(inputRangestereo,id);
      }
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
  
