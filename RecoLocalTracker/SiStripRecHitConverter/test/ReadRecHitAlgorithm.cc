// File: ReadRecHitAlgorithm.cc
// Description:  Analyzer that reads rechits
// Author:  C. Genta
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <algorithm>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/test/ReadRecHitAlgorithm.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;

ReadRecHitAlgorithm::ReadRecHitAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

ReadRecHitAlgorithm::~ReadRecHitAlgorithm() {
}


void ReadRecHitAlgorithm::run(const SiStripRecHit2DCollection* input)
{
  
  for (SiStripRecHit2DCollection::const_iterator detunit_iterator = input->begin(); detunit_iterator != input->end(); ++detunit_iterator) { 
      for(SiStripRecHit2DCollection::DetSet::const_iterator iter=detunit_iterator->begin(), end = detunit_iterator->end(); iter != end; ++iter){//loop on the rechit
          SiStripRecHit2D const & rechit=*iter;
          LocalPoint position=rechit.localPosition();
          LocalError error=rechit.localPositionError();
          //GeomDet& det=rechit->det();
          //DetId id=rechit.geographicalId();
          SiStripRecHit2D::ClusterRef clust=rechit.cluster();
          edm::LogInfo("ReadRecHit")<<"local position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<"\n"
              <<"local error: "<<error.xx()<<" "<<error.xy()<<" "<<error.yy();
          if (clust.isNonnull ()){
              edm::LogInfo("ReadRecHit")<<"barycenter= "<<clust->barycenter();
          }
          else{
              edm::LogError("ReadRecHit")<<"The cluster is empty!";
          }
          //	  std::cout<<"det id: "<<id.rawid<<std::endl;
      }
  }
}


void ReadRecHitAlgorithm::run(const SiStripMatchedRecHit2DCollection* input)
{
  
  for (SiStripMatchedRecHit2DCollection::const_iterator detunit_iterator = input->begin(); detunit_iterator != input->end(); ++detunit_iterator) { 
     for(SiStripMatchedRecHit2DCollection::DetSet::const_iterator iter=detunit_iterator->begin(), end = detunit_iterator->end(); iter != end; ++iter){//loop on the rechit
	  SiStripMatchedRecHit2D const & rechit=*iter;
	  LocalPoint position=rechit.localPosition();
	  LocalError error=rechit.localPositionError();
	  auto det=rechit.det();
	  //DetId id=rechit.geographicalId();
	  //	  std::vector<const SiStripCluster*> clust=rechit.cluster();
	  edm::LogInfo("ReadRecHit")<<"local position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<"\n"
	  <<"local error: "<<error.xx()<<" "<<error.xy()<<" "<<error.yy();

          auto m = iter->monoHit();
          auto s = iter->stereoHit();
	  ProjectedSiStripRecHit2D projrechit(m.localPosition() ,m.localPositionError() , *det, m );
	  ProjectedSiStripRecHit2D projsrechit(s.localPosition() ,s.localPositionError() , *det, s );
	  
	  edm::LogVerbatim("ReadRecHit")<<"Checking shareinput\nALL:";
	  
	  edm::LogVerbatim("ReadRecHit")<<"Matched with its mono ---------->";
	  if (iter->sharesInput(&m,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Matched with its stereo ---------->";
	  if (iter->sharesInput(&s,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Mono with its matched ---------->";
	  if (m.sharesInput(&rechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Stereo with its matched ---------->";
	  if (s.sharesInput(&rechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Mono with itsself ---------->";
	  if (m.sharesInput(&m,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Stereo with itsself ---------->";
	  if (s.sharesInput(&s,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Mono with stereo ---------->";
	  if (m.sharesInput(&s,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Stereo with mono ---------->";
	  if (s.sharesInput(&m,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";
  
       	  edm::LogVerbatim("ReadRecHit")<<"Mono with projected mono---------->";
	  if (m.sharesInput(&projrechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Stereo with projected mono ---------->";
	  if (s.sharesInput(&projrechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Matched with projected mono ---------->";
	  if (iter->sharesInput(&projrechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with mono---------->";
	  if (projrechit.sharesInput(&m,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with stereo ---------->";
	  if (projrechit.sharesInput(&s,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with matched ---------->";
	  if (projrechit.sharesInput(&rechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with projected stereo ---------->";
	  if (projrechit.sharesInput(&projsrechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with itsself ---------->";
	  if (projrechit.sharesInput(&projrechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Matched with itsself ---------->";
	  if (rechit.sharesInput(&rechit,TrackingRecHit::all))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Checking shareinput\nSOME:";
	  
	  edm::LogVerbatim("ReadRecHit")<<"Matched with its mono ---------->";
	  if (iter->sharesInput(&m,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Matched with its stereo ---------->";
	  if (iter->sharesInput(&s,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Mono with its matched ---------->";
	  if (m.sharesInput(&rechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Stereo with its matched ---------->";
	  if (s.sharesInput(&rechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Mono with itsself ---------->";
	  if (m.sharesInput(&m,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Stereo with itsself ---------->";
	  if (s.sharesInput(&s,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Mono with stereo ---------->";
	  if (m.sharesInput(&s,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Stereo with mono ---------->";
	  if (s.sharesInput(&m,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";
  
       	  edm::LogVerbatim("ReadRecHit")<<"Mono with projected mono---------->";
	  if (m.sharesInput(&projrechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Stereo with projected mono ---------->";
	  if (s.sharesInput(&projrechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Matched with projected mono ---------->";
	  if (iter->sharesInput(&projrechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with mono---------->";
	  if (projrechit.sharesInput(&m,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with stereo ---------->";
	  if (projrechit.sharesInput(&s,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with matched ---------->";
	  if (projrechit.sharesInput(&rechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with projected stereo ---------->";
	  if (projrechit.sharesInput(&projsrechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"FAILED!";
	  else edm::LogVerbatim("ReadRecHit")<<"PASSED!";

	  edm::LogVerbatim("ReadRecHit")<<"Projected mono with itsself ---------->";
	  if (projrechit.sharesInput(&projrechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

	  edm::LogVerbatim("ReadRecHit")<<"Matched with itsself ---------->";
	  if (rechit.sharesInput(&rechit,TrackingRecHit::some))edm::LogVerbatim("ReadRecHit")<<"PASSED!";
	  else edm::LogVerbatim("ReadRecHit")<<"FAILED!";

      }
  }
}




  
