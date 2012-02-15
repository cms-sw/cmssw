// -*- C++ -*-
//
// Class:      DTandCSCSegmentsinTracks
// 
/**\class DTandCSCSegmentsinTracks DTandCSCSegmentsinTracks.cc RecoLocalMuon/RPCRecHit/src/DTandCSCSegmentsinTracks.cc
   
   Description: [one line class summary]
   
   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Juan Pablo Gomez Cardona,42 R-023,+41227662349,
//         Created:  Thu Jan  19 13:03:28 CEST 2012
// $Id$
//
//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalMuon/RPCRecHit/interface/DTandCSCSegmentsinTracks.h"
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


using namespace std;

#include <memory>
#include <ctime>



DTandCSCSegmentsinTracks::DTandCSCSegmentsinTracks(const edm::ParameterSet& iConfig)
{
  cscSegments=iConfig.getParameter<edm::InputTag>("cscSegments");
  dt4DSegments=iConfig.getParameter<edm::InputTag>("dt4DSegments");
  tracks=iConfig.getParameter<edm::InputTag>("tracks");



  produces<DTRecSegment4DCollection>("SelectedDtSegments");
  produces<CSCSegmentCollection>("SelectedCscSegments");
  
  
  
}


DTandCSCSegmentsinTracks::~DTandCSCSegmentsinTracks(){
  
}

void DTandCSCSegmentsinTracks::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  iEvent.getByLabel(dt4DSegments, all4DSegments);	
  
  edm::Handle<CSCSegmentCollection> allCSCSegments;
  iEvent.getByLabel(cscSegments, allCSCSegments);
  
  edm::Handle<reco::TrackCollection> alltracks;
  iEvent.getByLabel(tracks,alltracks);
  
  reco::TrackCollection::const_iterator Track;
  DTRecSegment4DCollection::const_iterator segmentDT;
  CSCSegmentCollection::const_iterator segmentCSC;
  DTRecSegment4D segmentDTToStore;
  CSCSegment segmentCSCToStore;
  
  std::vector<int> positionDTSeg;
  std::vector<int> positionCSCSeg;
  
  std::auto_ptr<DTRecSegment4DCollection> selectedDtSegments(new DTRecSegment4DCollection());
  std::auto_ptr<CSCSegmentCollection> selectedCscSegments(new CSCSegmentCollection());
  
  DTRecSegment4DCollection::id_iterator chamberId;
  std::cout<<"paso1"<<std::endl; 
  //Loop over the chambers 
  for (chamberId = all4DSegments->id_begin(); chamberId != all4DSegments->id_end(); ++chamberId)
    {
      bool AcceptedDTSegment = false;
      edm::OwnVector<DTRecSegment4D> DTSegmentsVector;
      positionDTSeg.clear();
      std::cout<<"paso2"<<std::endl; 
      //loop over alltracks
      for  (Track=alltracks->begin(); Track!=alltracks->end();   Track++)  {
	std::cout<<"paso3"<<std::endl;
	//Loop over the hits of the track
	for( unsigned int counter = 0; counter != (*Track).recHitsSize()-1; counter++ )
	  {
	    TrackingRecHitRef myRef =(*Track).recHit( counter );
	    const TrackingRecHit *rechit = myRef.get();
	    const GeomDet* geomDet = theTrackingGeometry->idToDet( rechit->geographicalId() );
	    LocalPoint posTrackRecHit = rechit->localPosition();//N
	    std::cout<<"paso4"<<std::endl;
	    //It's a DT Hit
	    if( geomDet->subDetector() == GeomDetEnumerators::DT )
	      {
		//Take the layer associated to this hit
		DTLayerId myLayer( rechit->geographicalId().rawId() );
		std::cout<<"paso5"<<std::endl; 
		//If the layer of the hit belongs to the chamber of the 4D Segment
		if( myLayer.wheel() == (*chamberId).wheel() &&  myLayer.station() == (*chamberId).station() && myLayer.sector() == (*chamberId).sector() )
		  {
		    int NumberOfDTSegment = 0;
		    // Get the range for the corresponding ChamberId
		    DTRecSegment4DCollection::range  range = all4DSegments->get(*chamberId);
                    int NumberOfSegments = 0;
                    int NumberOfDTSegmentToStore = 0;  
		    std::cout<<"paso6"<<std::endl;
		    // Loop over the 4Dsegments of this ChamberId
		    for (segmentDT = range.first; segmentDT!=range.second; ++segmentDT) 
		      {
			//By default the chamber associated to the segment is new  
			bool isNewSegment = true;
			std::cout<<"paso7"<<std::endl;
			//Loop over 4Dsegments already included in the vector of segments 
			for( std::vector<int>::iterator positionItSeg = positionDTSeg.begin();
			     positionItSeg != positionDTSeg.end(); positionItSeg++ )
			  {
			    std::cout<<"Number of DT: "<<NumberOfDTSegment<<" , Number of the DTAccepted: "<<(*positionItSeg)<<std::endl;
			    //If this segment has been used before isNewChamber = false
			    if( NumberOfDTSegment == *positionItSeg ) 
			      {
				isNewSegment = false;
			      }
			  }//Loop over 4Dsegments already included in the vector of segments 
			std::cout<<"paso8"<<std::endl;
			//If the segment is new
			if( isNewSegment )
			  {
			    vector <const TrackingRecHit*> segments2D = (&(*segmentDT))->recHits();
			    // container for 4D segment recHits
			    vector <const TrackingRecHit*> dtRecHits;
			    std::cout<<"paso9"<<std::endl;
			    // Loop over the 2DSegments of this 4DSegment
			    for(vector<const TrackingRecHit*>::const_iterator segm2D = segments2D.begin(); segm2D != segments2D.end(); segm2D++)
			      {                          
				vector <const TrackingRecHit*> rHits1D = (*segm2D)->recHits();
				std::cout<<"paso10"<<std::endl;
				//loop over recHits of this 2DSegment
				for (int hit=0; hit<int(rHits1D.size()); hit++)
				  {
				    dtRecHits.push_back(rHits1D[hit]);
				  }//loop over recHits of this 2DSegment
			      }//Loop over the 2DSegments of this 4DSegment
			    //now for each 4Dsegment we have a vector of RecHit
			    //we loop over the recHit checking if there's the recHit of the track
			    std::cout<<"paso11"<<std::endl;
			    int NumberOfEqualRecHits=0; 
			    //loop over 4Dsegment recHits vector
			    for (unsigned int hit = 0; hit < dtRecHits.size(); hit++)
			      {                  
				LocalPoint posSegDTRecHit = (*dtRecHits[hit]).localPosition();
				double rDT=sqrt(pow((posSegDTRecHit.x()-posTrackRecHit.x()),2) +pow((posSegDTRecHit.y()-posTrackRecHit.y()),2) + pow((posSegDTRecHit.z()-posTrackRecHit.z()),2));                          
				if (rDT<0.0001)
				  {
				    NumberOfEqualRecHits++;  
				  }
			      }//loop over 4Dsegment recHits vector
			    //Common rechits?
                            if (0 < NumberOfEqualRecHits)
			      {      
				NumberOfSegments++;
                                segmentDTToStore = *segmentDT;
                                NumberOfDTSegmentToStore = NumberOfDTSegment;                                
			      }//Common rechits?			
			  }//If the segment is new
			std::cout<<"paso13"<<std::endl;
			std::cout<<" Seg: "<<NumberOfDTSegment<<std::endl;
			std::cout<<"achamber id: "<<(*chamberId)<<std::endl;
			NumberOfDTSegment++;
		      }// Loop over the 4Dsegments of this ChamberId
		    //Is just one segment?
		    if (NumberOfSegments == 1)
                      {
                        //push position of the segment and tracking rechit
			std::cout<<" Seg aceptado: "<<NumberOfDTSegmentToStore<<std::endl;
			AcceptedDTSegment = true;
			positionDTSeg.push_back( NumberOfDTSegmentToStore );//SolChamb
			DTSegmentsVector.push_back(segmentDTToStore); 
                      }//Is just one segment?
		    std::cout<<"paso15"<<std::endl;
		    
		  }//If the layer of the hit belongs to the chamber of the 4D Segment
		
	      }//It's a DT Hit
	    
	  }//Loop over the hits of the track    
      }//Loop over the tracks 
      if (AcceptedDTSegment){
	std::cout<<"paso16"<<std::endl;
	std::cout<<"chamber id: "<<(*chamberId)<<std::endl;
	selectedDtSegments->put(*chamberId, DTSegmentsVector.begin(), DTSegmentsVector.end());
	std::cout<<"paso17"<<std::endl;
      }
    }//Loop over the chambers
  
  
  
  
  CSCSegmentCollection::id_iterator chamberIdCSC;
  //Loop over the chambers 
  for (chamberIdCSC = allCSCSegments->id_begin(); chamberIdCSC != allCSCSegments->id_end(); ++chamberIdCSC)
    {
      bool AcceptedCSCSegment = false;
      edm::OwnVector<CSCSegment> CSCSegmentsVector;
      positionCSCSeg.clear();
      std::cout<<"paso2"<<std::endl; 
      //loop over alltracks
      for  (Track=alltracks->begin(); Track!=alltracks->end();   Track++)  
	{
	  std::cout<<"paso3"<<std::endl;
	  //Loop over the hits of the track
	  for( unsigned int counter = 0; counter != (*Track).recHitsSize()-1; counter++ )
	    {
	      TrackingRecHitRef myRef =(*Track).recHit( counter );
	      const TrackingRecHit *rechit = myRef.get();
	      const GeomDet* geomDet = theTrackingGeometry->idToDet( rechit->geographicalId() );
	      LocalPoint posTrackRecHit = rechit->localPosition();//N
	      std::cout<<"paso4"<<std::endl;
	      //It's a CSC Hit
	      if ( geomDet->subDetector() == GeomDetEnumerators::CSC )
		{
		  //Take the layer associated to this hit
		  CSCDetId myLayer( rechit->geographicalId().rawId() );
		  std::cout<<"paso5"<<std::endl; 
		  //If the layer of the hit belongs to the chamber of the Segment
		  if( myLayer.chamberId() == (*chamberIdCSC).chamberId() )
		    {
		      int NumberOfCSCSegment = 0;
		      // Get the range for the corresponding ChamerId                                                                                                           
		      CSCSegmentCollection::range  range = allCSCSegments->get(*chamberIdCSC);
		      int NumberOfSegments = 0;
		      int NumberOfCSCSegmentToStore = 0;  
		      std::cout<<"paso6"<<std::endl;
		      // Loop over the rechits of this ChamberId                                                                                                                
		      for (segmentCSC = range.first;
			   segmentCSC!=range.second; ++segmentCSC) {
			//By default the chamber associated to the segment is new
			bool isNewSegment = true;
			std::cout<<"paso7"<<std::endl;
			//Loop over segments already include in the vector of segments
			for( std::vector<int>::iterator positionItSeg = positionCSCSeg.begin(); positionItSeg != positionCSCSeg.end(); positionItSeg++ )
			  {
			    std::cout<<"Number of CSC: "<<NumberOfCSCSegment<<" , Number of the CSCAccepted: "<<(*positionItSeg)<<std::endl;
			    //If this segment has been used before isNewChamber = false
			    if(NumberOfCSCSegment == *positionItSeg)
			      {
				isNewSegment = false;
			      }
			  }//Loop over segments already include in the vector of segments		
			std::cout<<"paso8"<<std::endl;
			//If the segment is new
			if( isNewSegment )
			  {
			    
			    // container for segment recHits
			    vector <const TrackingRecHit*> rHits = (&(*segmentCSC))->recHits();
			    std::cout<<"paso10"<<std::endl;
			    //we loop over the recHit checking if there's the recHit of the track
			    int NumberOfEqualRecHits=0; 
			    //loop over segment recHits vector
			    for (unsigned int hit = 0; hit < rHits.size(); hit++)
			      {                  
				LocalPoint posSegCSCRecHit = (*rHits[hit]).localPosition();
				double rCSC=sqrt(pow((posSegCSCRecHit.x()-posTrackRecHit.x()),2) +pow((posSegCSCRecHit.y()-posTrackRecHit.y()),2) + pow((posSegCSCRecHit.z()-posTrackRecHit.z()),2));                          
				if (rCSC<0.0001)
				  {
				    NumberOfEqualRecHits++;  
				  }
			      }//loop over segment recHits vector
			    //Common rechits?
			    if (0 < NumberOfEqualRecHits)
			      {      
				NumberOfSegments++;
				segmentCSCToStore = *segmentCSC;
				NumberOfCSCSegmentToStore = NumberOfCSCSegment;                                
			      }//Common rechits?			
			  }//If the segment is new
			std::cout<<"paso13"<<std::endl;
			std::cout<<" Seg: "<<NumberOfCSCSegment<<std::endl;
			std::cout<<"achamber id: "<<(*chamberIdCSC)<<std::endl;
			NumberOfCSCSegment++;
		      }// Loop over the 4Dsegments of this ChamberId
		      //Is just one segment?
		      if (NumberOfSegments == 1)
			{
			  //push position of the segment and tracking rechit
			  std::cout<<" Seg aceptado: "<<NumberOfCSCSegmentToStore<<std::endl;
			  AcceptedCSCSegment = true;
			  positionCSCSeg.push_back( NumberOfCSCSegmentToStore );
			  CSCSegmentsVector.push_back(segmentCSCToStore); 
			}//Is just one segment?
		      std::cout<<"paso15"<<std::endl;
		      
		    }//If the layer of the hit belongs to the chamber of the Segment
		  
		}//It's a CSC Hit
	      
	    }//Loop over the hits of the track    
	}//Loop over the tracks 
      if (AcceptedCSCSegment){
	std::cout<<"paso16"<<std::endl;
	std::cout<<"chamber id: "<<(*chamberIdCSC)<<std::endl;
	selectedCscSegments->put(*chamberIdCSC, CSCSegmentsVector.begin(), CSCSegmentsVector.end());
	std::cout<<"paso17"<<std::endl;
      }
    }//Loop over the chambers
  
  
  
  
  std::cout<<"paso201"<<std::endl;
  iEvent.put(selectedCscSegments,"SelectedCscSegments");
  iEvent.put(selectedDtSegments,"SelectedDtSegments");
  std::cout<<"paso202"<<std::endl;
}


// ------------ method called once each job just before starting event loop  ------------
void 
DTandCSCSegmentsinTracks::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DTandCSCSegmentsinTracks::endJob() {
}

DEFINE_FWK_MODULE(DTandCSCSegmentsinTracks);
