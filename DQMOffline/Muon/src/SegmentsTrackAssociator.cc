

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/05/07 14:05:09 $
 *  $Revision: 1.2 $
 *  \author C. Botta - INFN Torino
 *  Revised by G. Mila
 */


#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"



#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DQMOffline/Muon/src/SegmentsTrackAssociator.h"


#include <vector>

using namespace edm;
using namespace std;



SegmentsTrackAssociator::SegmentsTrackAssociator(const ParameterSet& iConfig)
{
  theDTSegmentLabel = iConfig.getUntrackedParameter<InputTag>("segmentsDt");
  theCSCSegmentLabel = iConfig.getUntrackedParameter<InputTag>("segmentsCSC");
  theSegmentContainerName = iConfig.getUntrackedParameter<InputTag>("SelectedSegments");
    
  NrecHit=0;
  NrecHitDT=0;
  NrecHitCSC=0;
  metname = "SegmentsTrackAssociator";
}


SegmentsTrackAssociator::~SegmentsTrackAssociator();


MuonTransientTrackingRecHit::MuonRecHitContainer SegmentsTrackAssociator::associate(const Event& iEvent, const EventSetup& iSetup, const reco::Track& Track){

  // The segment collections
  Handle<DTRecSegment4DCollection> DTSegments;
  iEvent.getByLabel(theDTSegmentLabel, DTSegments); 
  Handle<CSCSegmentCollection> CSCSegments;
  iEvent.getByLabel(theCSCSegmentLabel, CSCSegments);
  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  

  DTRecSegment4DCollection::const_iterator segment;
  CSCSegmentCollection::const_iterator segment2;
  MuonTransientTrackingRecHit::MuonRecHitContainer SelectedSegments;


  //loop over recHit
  for(trackingRecHit_iterator recHit =  Track.recHitsBegin(); recHit != Track.recHitsEnd(); ++recHit){
    
    NrecHit++;
 
    //get the detector Id and the local position
    DetId IdRivHit = (*recHit)->geographicalId();
    LocalPoint posTrackRecHit = (*recHit)->localPosition(); 
    
    
    // for DT recHits
    if (IdRivHit.subdetId() == MuonSubdetId::DT ) {

      DTRecSegment4DCollection::range range; 	
      NrecHitDT++;

      // get the chamber Id
      DTChamberId chamberId(IdRivHit.rawId());
      // get the segments of the chamber
      range = DTSegments->get(chamberId);
      //loop over segments
      for (segment = range.first; segment!=range.second; segment++){

	DetId Id = segment->geographicalId();
	const GeomDet* det = theTrackingGeometry->idToDet(Id);
	
	vector<const TrackingRecHit*> segments2D = (&(*segment))->recHits();
	// container for 4D segment recHits
	vector <const TrackingRecHit*> DTrecHits;

	for(vector<const TrackingRecHit*>::const_iterator segm2D = segments2D.begin();
	  segm2D != segments2D.end();
	  segm2D++) {

	  vector <const TrackingRecHit*> rHits1D = (*segm2D)->recHits();
	  for (int hit=0; hit<int(rHits1D.size()); hit++){
	    DTrecHits.push_back(rHits1D[hit]);
	  }

	}
	
	// loop over the recHit checking if there's the recHit of the track
	for (unsigned int hit = 0; hit < DTrecHits.size(); hit++) { 	  
	  
	  DetId IdRivHitSeg = (*DTrecHits[hit]).geographicalId();
	  LocalPoint posDTSegment=  segment->localPosition();
	  LocalPoint posSegDTRecHit = (*DTrecHits[hit]).localPosition(); 
	  	    
	  double rDT=sqrt(pow((posSegDTRecHit.x()-posTrackRecHit.x()),2) +pow((posSegDTRecHit.y()-posTrackRecHit.y()),2) + pow((posSegDTRecHit.z()-posTrackRecHit.z()),2));
	  
	  if (IdRivHit == IdRivHitSeg && rDT<0.0001){  

	    if (SelectedSegments.empty()){
		SelectedSegments.push_back(MuonTransientTrackingRecHit::specificBuild(det,&*segment));
		LogTrace(metname) <<"First selected segment (from DT). Position : "<<posDTSegment<<"  Chamber : "<<segment->chamberId();
	    }
	    else{
	      int check=0;
	      for(int segm=0; segm < int(SelectedSegments.size()); ++segm) {
		double dist=sqrt(pow((((*(SelectedSegments[segm])).localPosition()).x()-posDTSegment.x()),2) +pow((((*(SelectedSegments[segm])).localPosition()).y()-posDTSegment.y()),2) + pow((((*(SelectedSegments[segm])).localPosition()).z()-posDTSegment.z()),2));
		if(dist>0.000001) check++;
	      }     
		
	      if(check==int(SelectedSegments.size())){
		SelectedSegments.push_back(MuonTransientTrackingRecHit::specificBuild(det,&*segment));
		LogTrace(metname) <<"New DT selected segment. Position : "<<posDTSegment<<"  Chamber : "<<segment->chamberId();
	      }      
	    }	
	  } // check to tag the segment as "selected"

	} // loop over segment recHits

      } // loop over DT segments
    }
    
    
   
    // for CSC recHits
    if (IdRivHit.subdetId() == MuonSubdetId::CSC ) {

      CSCSegmentCollection::range range; 
      NrecHitCSC++;

      // get the chamber Id
      CSCDetId tempchamberId(IdRivHit.rawId());
      
      int ring = tempchamberId.ring();
      int station = tempchamberId.station();
      int endcap = tempchamberId.endcap();
      int chamber = tempchamberId.chamber();	
      CSCDetId chamberId(endcap, station, ring, chamber, 0);
      
      // get the segments of the chamber
      range = CSCSegments->get(chamberId);
      // loop over segments
      for(segment2 = range.first; segment2!=range.second; segment2++){

	DetId Id2 = segment2->geographicalId();
	const GeomDet* det2 = theTrackingGeometry->idToDet(Id2);
	
	// container for CSC segment recHits
	vector<const TrackingRecHit*> CSCrecHits = (&(*segment2))->recHits();
	
	// loop over the recHit checking if there's the recHit of the track	
	for (unsigned int hit = 0; hit < CSCrecHits.size(); hit++) { 
  
	  DetId IdRivHitSeg = (*CSCrecHits[hit]).geographicalId();
	  LocalPoint posSegCSCRecHit = (*CSCrecHits[hit]).localPosition(); 
	  LocalPoint posCSCSegment=  segment2->localPosition();
	    	  
	  double rCSC=sqrt(pow((posSegCSCRecHit.x()-posTrackRecHit.x()),2) +pow((posSegCSCRecHit.y()-posTrackRecHit.y()),2) + pow((posSegCSCRecHit.z()-posTrackRecHit.z()),2));

	  if (IdRivHit==IdRivHitSeg && rCSC < 0.0001){

	    if (SelectedSegments.empty()){
	      SelectedSegments.push_back(MuonTransientTrackingRecHit::specificBuild(det2,&*segment2));
	      LogTrace(metname) <<"First selected segment (from CSC). Position: "<<posCSCSegment;
	    }
	    else{
	      int check=0;
	      for(int n=0; n< int(SelectedSegments.size()); n++){
		double dist = sqrt(pow(((*(SelectedSegments[n])).localPosition().x()-posCSCSegment.x()),2) +pow(((*(SelectedSegments[n])).localPosition().y()-posCSCSegment.y()),2) + pow(((*(SelectedSegments[n])).localPosition().z()-posCSCSegment.z()),2));
		if(dist>0.000001) check++;
	      }
	      if(SelectedSegments.size()){  
		SelectedSegments.push_back(MuonTransientTrackingRecHit::specificBuild(det2,&*segment2));
		LogTrace(metname) <<"New CSC segment selected. Position : "<<posCSCSegment;	
	      }
	    }
	    
	  } // check to tag the segment as "selected"
	    
	} // loop over segment recHits
	
      } // loop over DT segments
    } 
      
  } // loop over track recHits
    
  LogTrace(metname) <<"N recHit:"<<NrecHit;
  NrecHit=0;
  LogTrace(metname) <<"N recHit DT:"<<NrecHitDT;
  NrecHitDT=0;
  LogTrace(metname) <<"N recHit CSC:"<<NrecHitCSC;
  NrecHitCSC=0;
  LogTrace(metname) <<"N selected segments:"<<SelectedSegments.size();
  
  return SelectedSegments;
  
}
