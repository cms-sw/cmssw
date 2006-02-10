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
  clustermatch_=new SiStripRecHitMatcher();
}

SiStripRecHitConverterAlgorithm::~SiStripRecHitConverterAlgorithm() {
  if(clustermatch_!=0){
    delete clustermatch_;
  }
}
void SiStripRecHitConverterAlgorithm::run(const SiStripClusterCollection* input,SiStripRecHit2DMatchedLocalPosCollection & outmatched,SiStripRecHit2DLocalPosCollection & outrphi, SiStripRecHit2DLocalPosCollection & outstereo,const TrackingGeometry& tracker)
{
  run(input, outmatched,outrphi,outstereo,tracker,LocalVector(0.,0.,0.));
}


void SiStripRecHitConverterAlgorithm::run(const SiStripClusterCollection* input,SiStripRecHit2DMatchedLocalPosCollection & outmatched,SiStripRecHit2DLocalPosCollection & outrphi, SiStripRecHit2DLocalPosCollection & outstereo,const TrackingGeometry& tracker,LocalVector trackdirection)
{
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
      const GeomDetUnit * stripdet=tracker.idToDet(detId);
      const SiStripClusterCollection::Range clusterRange = input->get(id);
      SiStripClusterCollection::ContainerIterator clusterRangeIteratorBegin = clusterRange.first;
      SiStripClusterCollection::ContainerIterator clusterRangeIteratorEnd   = clusterRange.second;
      SiStripClusterCollection::ContainerIterator iter;
      StripSubdetector specDetId=(StripSubdetector)(*detunit_iterator);
      if(detId.subdetId()==StripSubdetector::TIB||detId.subdetId()==StripSubdetector::TOB){// if TIB of TOB use the rectangualr strip topology 
	const RectangularStripTopology& Rtopol=(RectangularStripTopology&)stripdet->topology();
	// get the partner id only if the detector is r-phi
	for(iter=clusterRangeIteratorBegin;iter!=clusterRangeIteratorEnd;++iter){//loop on the cluster
	  //SiStripCluster cluster=*iter;
	  LocalPoint position=Rtopol.localPosition(iter->barycenter());
	  const  LocalError dummy;
	  std::vector<const SiStripCluster*> clusters;
	  clusters.push_back(&(*iter));
	  if(!specDetId.stereo()){
	    collectorrphi.push_back(new SiStripRecHit2DLocalPos(position, dummy,detId,clusters));
	    nmono++;
	  }
	  if(specDetId.stereo()){
	    collectorstereo.push_back(new SiStripRecHit2DLocalPos(position, dummy,detId,clusters));
	    nstereo++;
	  }
	}
      } else if (detId.subdetId()==(StripSubdetector::TID)||detId.subdetId()==(StripSubdetector::TEC)){    //if TID or TEC use trapoeziodalstrip topology
	const TrapezoidalStripTopology& Ttopol=(TrapezoidalStripTopology&)stripdet->topology();
	// get the partner id only if the detector is r-phi
    	for(iter=clusterRangeIteratorBegin;iter!=clusterRangeIteratorEnd;++iter){//loop on the cluster
	  //SiStripCluster cluster=*iter;
	  LocalPoint position=Ttopol.localPosition(iter->barycenter());
	  const  LocalError dummy;
	  std::vector<const SiStripCluster*> clusters;
	  clusters.push_back(&(*iter));
	  if(!specDetId.stereo()){
	    collectorrphi.push_back(new SiStripRecHit2DLocalPos(position, dummy,detId,clusters));
	    nmono++;
	  }
	  if(specDetId.stereo()){
	    collectorstereo.push_back(new SiStripRecHit2DLocalPos(position, dummy,detId,clusters));
	    nstereo++;
	  }
	}
      } 
      SiStripRecHit2DLocalPosCollection::Range inputRangerphi(collectorrphi.begin(),collectorrphi.end());
      SiStripRecHit2DLocalPosCollection::Range inputRangestereo(collectorstereo.begin(),collectorstereo.end());
      
      if (collectorrphi.size() > 0) {
	outrphi.put(inputRangerphi,id);
      }
      if (collectorstereo.size() > 0) {
	outstereo.put(inputRangestereo,id);
      }
    }
  }
//
// match the clusters
//
  
  const std::vector<unsigned int> detIDs2 = outrphi.detIDs();
  for ( std::vector<unsigned int>::const_iterator detunit_iterator = detIDs2.begin(); detunit_iterator != detIDs2.end(); detunit_iterator++ ) {//loop over detectors
    edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collectorMatched; 
    SiStripRecHit2DLocalPosCollection::Range monoRecHitRange = outrphi.get(*detunit_iterator);
    SiStripRecHit2DLocalPosCollection::ContainerConstIterator rhRangeIteratorBegin = monoRecHitRange.first;
    SiStripRecHit2DLocalPosCollection::ContainerConstIterator rhRangeIteratorEnd   = monoRecHitRange.second;
    SiStripRecHit2DLocalPosCollection::ContainerConstIterator iter;
    unsigned int id = 0;
    for(iter=rhRangeIteratorBegin;iter!=rhRangeIteratorEnd;++iter){//loop on the mono RH
      edm::OwnVector<SiStripRecHit2DMatchedLocalPos> collectorMatchedSingleHit; 
      StripSubdetector specDetId(*detunit_iterator);
      id = specDetId.partnerDetId();
      std::vector<unsigned int>::const_iterator partnerdetiter=std::find(detIDs.begin(),detIDs.end(),id);
      if(partnerdetiter==detIDs.end()) id=0;	
      if (id>0){
	DetId partnerdetId(id);
	const GeomDetUnit * monostripdet=tracker.idToDet(DetId(*detunit_iterator));
	const GeomDetUnit * stereostripdet=tracker.idToDet(DetId(id));
	
	const SiStripRecHit2DLocalPosCollection::Range rhpartnerRange = outstereo.get(id);
	SiStripRecHit2DLocalPosCollection::ContainerConstIterator rhpartnerRangeIteratorBegin = rhpartnerRange.first;
	SiStripRecHit2DLocalPosCollection::ContainerConstIterator rhpartnerRangeIteratorEnd   = rhpartnerRange.second;
	
	//	edm::OwnVector<SiStripRecHit2DMatchedLocalPos> tempCollector; 
	
	const DetId theId(id);

	if(partnerdetId.subdetId()==StripSubdetector::TIB||partnerdetId.subdetId()==StripSubdetector::TOB){// if TIB of TOB use the rectangualr strip topology 
	  const RectangularStripTopology& Rtopol=(RectangularStripTopology&)stereostripdet->topology();
	  collectorMatchedSingleHit=clustermatch_->match<RectangularStripTopology>(&(*iter),rhpartnerRangeIteratorBegin,rhpartnerRangeIteratorEnd,theId,Rtopol,monostripdet,stereostripdet,trackdirection);
	}else{
	  const TrapezoidalStripTopology& Ttopol=(TrapezoidalStripTopology&)stereostripdet->topology();
	  collectorMatchedSingleHit=clustermatch_->match<TrapezoidalStripTopology>(&(*iter),rhpartnerRangeIteratorBegin,rhpartnerRangeIteratorEnd,theId,Ttopol,monostripdet,stereostripdet,trackdirection);
	}
	if (collectorMatchedSingleHit.size()>0) {
	  nmatch++;
	}else{
	  nunmatch++;
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
    SiStripRecHit2DMatchedLocalPosCollection::Range inputRangematched(collectorMatched.begin(),collectorMatched.end());
    
    if (collectorMatched.size()>0){
      outmatched.put(inputRangematched, *detunit_iterator);
    }
    //    SiStripRecHit2DLocalPosCollection::Range inputRangematched(collectorMatched.begin(),collectorMatched.end());
    
    //    if (collectormatched.size() > 0) {
    //      outmatched.put(inputRangematched,id);
    //}

  }

  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    std::cout << "[SiStripRecHitConverterAlgorithm] found" << std::endl; 
    std::cout << nmono << "  clusters in mono detectors"<< std::endl;
    std::cout << nstereo << "  clusters in partners stereo detectors"<< std::endl;
    std::cout << nmatch << "  matched RecHit"<< std::endl;
    std::cout << nunmatch << "  unmatched clusters "<< std::endl;
  }
};
  
