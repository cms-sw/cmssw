

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/10/24 10:16:27 $
 *  $Revision: 1.4 $
 *  \author C. Botta, G. Mila - INFN Torino
 */


#include "RecoMuon/TrackingTools/interface/SegmentsTrackAssociator.h"

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

#include <vector>

using namespace edm;
using namespace std;



SegmentsTrackAssociator::SegmentsTrackAssociator(const ParameterSet& iConfig)
{
  theDTSegmentLabel = iConfig.getUntrackedParameter<InputTag>("segmentsDt");
  theCSCSegmentLabel = iConfig.getUntrackedParameter<InputTag>("segmentsCSC");
  theSegmentContainerName = iConfig.getUntrackedParameter<InputTag>("SelectedSegments");
    
  numRecHit=0;
  numRecHitDT=0;
  numRecHitCSC=0;
  metname = "SegmentsTrackAssociator";
}


SegmentsTrackAssociator::~SegmentsTrackAssociator() {}


MuonTransientTrackingRecHit::MuonRecHitContainer SegmentsTrackAssociator::associate(const Event& iEvent, const EventSetup& iSetup, const reco::Track& Track){

  // The segment collections
  Handle<DTRecSegment4DCollection> dtSegments;
  iEvent.getByLabel(theDTSegmentLabel, dtSegments); 
  Handle<CSCSegmentCollection> cscSegments;
  iEvent.getByLabel(theCSCSegmentLabel, cscSegments);
  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  

  DTRecSegment4DCollection::const_iterator segment;
  CSCSegmentCollection::const_iterator segment2;
  MuonTransientTrackingRecHit::MuonRecHitContainer selectedSegments;
  MuonTransientTrackingRecHit::MuonRecHitContainer selectedDtSegments;
  MuonTransientTrackingRecHit::MuonRecHitContainer selectedCscSegments;

  //loop over recHit
  for(trackingRecHit_iterator recHit =  Track.recHitsBegin(); recHit != Track.recHitsEnd(); ++recHit){

    if(!(*recHit)->isValid()) continue;

    
    numRecHit++;
 
    //get the detector Id
    DetId idRivHit = (*recHit)->geographicalId();
      

    // DT recHits
    if (idRivHit.det() == DetId::Muon && idRivHit.subdetId() == MuonSubdetId::DT ) {

      // get the RecHit Local Position
      LocalPoint posTrackRecHit = (*recHit)->localPosition(); 

      DTRecSegment4DCollection::range range; 	
      numRecHitDT++;

      // get the chamber Id
      DTChamberId chamberId(idRivHit.rawId());
      // get the segments of the chamber
      range = dtSegments->get(chamberId);

      // loop over segments
      for (segment = range.first; segment!=range.second; segment++){

	DetId id = segment->geographicalId();
	const GeomDet* det = theTrackingGeometry->idToDet(id);
	vector<const TrackingRecHit*> segments2D = (&(*segment))->recHits();
	// container for 4D segment recHits
	vector <const TrackingRecHit*> dtRecHits;

	for(vector<const TrackingRecHit*>::const_iterator segm2D = segments2D.begin();
	  segm2D != segments2D.end();
	  segm2D++) {

	  vector <const TrackingRecHit*> rHits1D = (*segm2D)->recHits();
	  for (int hit=0; hit<int(rHits1D.size()); hit++){
	    dtRecHits.push_back(rHits1D[hit]);
	  }

	}
	
	// loop over the recHit checking if there's the recHit of the track
	for (unsigned int hit = 0; hit < dtRecHits.size(); hit++) { 	  
	  
	  DetId idRivHitSeg = (*dtRecHits[hit]).geographicalId();
	  LocalPoint posDTSegment=  segment->localPosition();
	  LocalPoint posSegDTRecHit = (*dtRecHits[hit]).localPosition(); 
	  	    
	  double rDT=sqrt(pow((posSegDTRecHit.x()-posTrackRecHit.x()),2) +pow((posSegDTRecHit.y()-posTrackRecHit.y()),2) + pow((posSegDTRecHit.z()-posTrackRecHit.z()),2));
	  
	  if (idRivHit == idRivHitSeg && rDT<0.0001){  

	    if (selectedDtSegments.empty()){
		selectedDtSegments.push_back(MuonTransientTrackingRecHit::specificBuild(det,&*segment));
		LogTrace(metname) <<"First selected segment (from DT). Position : "<<posDTSegment<<"  Chamber : "<<segment->chamberId();
	    }
	    else{
	      int check=0;
	      for(int segm=0; segm < int(selectedDtSegments.size()); ++segm) {
		double dist=sqrt(pow((((*(selectedDtSegments[segm])).localPosition()).x()-posDTSegment.x()),2) +pow((((*(selectedDtSegments[segm])).localPosition()).y()-posDTSegment.y()),2) + pow((((*(selectedDtSegments[segm])).localPosition()).z()-posDTSegment.z()),2));
		if(dist>30) check++;
	      }     
		
	      if(check==int(selectedDtSegments.size())){
		selectedDtSegments.push_back(MuonTransientTrackingRecHit::specificBuild(det,&*segment));
		LogTrace(metname) <<"New DT selected segment. Position : "<<posDTSegment<<"  Chamber : "<<segment->chamberId();
	      }      
	    }	
	  } // check to tag the segment as "selected"

	} // loop over segment recHits

      } // loop over DT segments
    }
    
   
    // CSC recHits
    if (idRivHit.det() == DetId::Muon && idRivHit.subdetId() == MuonSubdetId::CSC ) {

      // get the RecHit Local Position
      LocalPoint posTrackRecHit = (*recHit)->localPosition(); 

      CSCSegmentCollection::range range; 
      numRecHitCSC++;

      // get the chamber Id
      CSCDetId tempchamberId(idRivHit.rawId());
      
      int ring = tempchamberId.ring();
      int station = tempchamberId.station();
      int endcap = tempchamberId.endcap();
      int chamber = tempchamberId.chamber();	
      CSCDetId chamberId(endcap, station, ring, chamber, 0);
      
      // get the segments of the chamber
      range = cscSegments->get(chamberId);
      // loop over segments
      for(segment2 = range.first; segment2!=range.second; segment2++){

	DetId id2 = segment2->geographicalId();
	const GeomDet* det2 = theTrackingGeometry->idToDet(id2);
	
	// container for CSC segment recHits
	vector<const TrackingRecHit*> cscRecHits = (&(*segment2))->recHits();
	
	// loop over the recHit checking if there's the recHit of the track	
	for (unsigned int hit = 0; hit < cscRecHits.size(); hit++) { 
  
	  DetId idRivHitSeg = (*cscRecHits[hit]).geographicalId();
	  LocalPoint posSegCSCRecHit = (*cscRecHits[hit]).localPosition(); 
	  LocalPoint posCSCSegment=  segment2->localPosition();
	    	  
	  double rCSC=sqrt(pow((posSegCSCRecHit.x()-posTrackRecHit.x()),2) +pow((posSegCSCRecHit.y()-posTrackRecHit.y()),2) + pow((posSegCSCRecHit.z()-posTrackRecHit.z()),2));

	  if (idRivHit==idRivHitSeg && rCSC < 0.0001){

	    if (selectedCscSegments.empty()){
	      selectedCscSegments.push_back(MuonTransientTrackingRecHit::specificBuild(det2,&*segment2));
	      LogTrace(metname) <<"First selected segment (from CSC). Position: "<<posCSCSegment;
	    }
	    else{
	      int check=0;
	      for(int n=0; n< int(selectedCscSegments.size()); n++){
		double dist = sqrt(pow(((*(selectedCscSegments[n])).localPosition().x()-posCSCSegment.x()),2) +pow(((*(selectedCscSegments[n])).localPosition().y()-posCSCSegment.y()),2) + pow(((*(selectedCscSegments[n])).localPosition().z()-posCSCSegment.z()),2));
		if(dist>30) check++;
	      }
	      if(check==int(selectedCscSegments.size())){  
		selectedCscSegments.push_back(MuonTransientTrackingRecHit::specificBuild(det2,&*segment2));
		LogTrace(metname) <<"New CSC segment selected. Position : "<<posCSCSegment;	
	      }
	    }
	    
	  } // check to tag the segment as "selected"
	    
	} // loop over segment recHits
	
      } // loop over DT segments
    } 
      
  } // loop over track recHits
    
  LogTrace(metname) <<"N recHit:"<<numRecHit;
  numRecHit=0;
  LogTrace(metname) <<"N recHit DT:"<<numRecHitDT;
  numRecHitDT=0;
  LogTrace(metname) <<"N recHit CSC:"<<numRecHitCSC;
  numRecHitCSC=0;
  
  copy(selectedDtSegments.begin(), selectedDtSegments.end(), back_inserter(selectedSegments));
  LogTrace(metname) <<"N selected Dt segments:"<<selectedDtSegments.size();
  copy(selectedCscSegments.begin(), selectedCscSegments.end(), back_inserter(selectedSegments));
  LogTrace(metname) <<"N selected Csc segments:"<<selectedCscSegments.size();
  LogTrace(metname) <<"N selected segments:"<<selectedSegments.size();

  return selectedSegments;
  
}
