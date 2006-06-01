
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTSegmentAnalysis.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "Geometry/Vector/interface/Pi.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include <iterator>

using namespace edm;
using namespace std;

DTSegmentAnalysis::DTSegmentAnalysis(const ParameterSet& pset,
				     DaqMonitorBEInterface* dbe) : theDbe(dbe) {
				       debug = pset.getUntrackedParameter<bool>("debug","false");
				       // the name of the 4D rec hits collection
				       theRecHits4DLabel = pset.getParameter<string>("recHits4DLabel");
				     }

DTSegmentAnalysis::~DTSegmentAnalysis(){}


void DTSegmentAnalysis::analyze(const Event& event, const EventSetup& setup) {
  if(debug)
    cout << "[DTSegmentAnalysis] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;


  // -- 4D segment analysis  -----------------------------------------------------
  // Get the 4D segment collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments);

  // Loop over all chambers containing a segment
  DTRecSegment4DCollection::id_iterator chamberId;
  for (chamberId = all4DSegments->id_begin();
       chamberId != all4DSegments->id_end();
       ++chamberId){
    // Get the range for the corresponding ChamerId
    DTRecSegment4DCollection::range  range = all4DSegments->get(*chamberId);
    int nsegm = distance(range.first, range.second);
    cout << "   Chamber: " << *chamberId << " has " << nsegm
	 << " 4D segments" << endl;
    //FIXME: fill histo about number of segments
    fillHistos(*chamberId, nsegm);
    // Loop over the rechits of this ChamerId
    for (DTRecSegment4DCollection::const_iterator segment4D = range.first;
	 segment4D!=range.second;
	   ++segment4D){
      LocalPoint segment4DLocalPos = (*segment4D).localPosition();
      LocalVector segment4DLocalDirection = (*segment4D).localDirection();
      
      if (segment4DLocalDirection.z()) {
	fillHistos(*chamberId,
		   segment4DLocalPos.x(), 
		   segment4DLocalPos.y(),
		   segment4DLocalDirection.x()/segment4DLocalDirection.z()* 180./Geom::pi(),
		   segment4DLocalDirection.y()/segment4DLocalDirection.z()* 180./Geom::pi(),
		   (*segment4D).chi2()/(*segment4D).degreesOfFreedom());
      } else {
	cout << "[DTSegmentAnalysis] Warning: segment local direction is: "
	     << segment4DLocalDirection << endl;
      }
    }
  }
  // -----------------------------------------------------------------------------

}
  

  
// Book a set of histograms for a give chamber
void DTSegmentAnalysis::bookHistos(DTChamberId chamberId) {
  if(debug)
    cout << "   Booking histos for chamber: " << chamberId << endl;

  // Compose the chamber name
  stringstream wheel; wheel << chamberId.wheel();	
  stringstream station; station << chamberId.station();	
  stringstream sector; sector << chamberId.sector();	
  //   stringstream superLayer; superLayer << chamberId.superlayer();	
  //   stringstream layer; layer << chamberId.layer();	
  
  string chamberHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();
    
  theDbe->setCurrentFolder("DT/DTLocalRecoTask/Wheel" + wheel.str() +
			   "/Station" + station.str() +
			   "/Sector" + sector.str() + "/");
  // Create the monitor elements
  histosPerCh[chamberId][0] = theDbe->book1D("hN4DSeg"+chamberHistoName,
					     "# of 4D segments per event",
					     100, 0, 100);
  histosPerCh[chamberId][1] = theDbe->book1D("h4DSegmXInCham"+chamberHistoName,
					     "4D Segment X position (cm) in Chamer RF",
					     200, -200, 200);
  histosPerCh[chamberId][2] = theDbe->book1D("h4DSegmYInCham"+chamberHistoName,
					     "4D Segment Y position (cm) in Chamer RF",
					     200, -200, 200);
  histosPerCh[chamberId][3] = theDbe->book2D("h4DSegmXvsYInCham"+chamberHistoName,
					     "4D Segment position (cm) in Chamer RF",
					     200, -200, 200, 200, -200, 200);
  histosPerCh[chamberId][4] = theDbe->book1D("h4DSegmPhiDirection"+chamberHistoName,
					     "4D Segment Phi Direction (deg)",
					     180, -180, 180);
  histosPerCh[chamberId][5] = theDbe->book1D("h4DSegm ThetaDirection"+chamberHistoName,
					     "4D Segment  Theta Direction (deg)",
					     180, -180, 180);
  histosPerCh[chamberId][6] = theDbe->book1D("h4DChi2"+chamberHistoName,
					     "4D Segment reduced Chi2",
					     30, 0, 30);
}



// Fill a set of histograms for a give chamber 
void DTSegmentAnalysis::fillHistos(DTChamberId chamberId, int nsegm) {
  // FIXME: optimization of the number of searches
  if(histosPerCh.find(chamberId) == histosPerCh.end()) {
    bookHistos(chamberId);
  }
  histosPerCh[chamberId][0]->Fill(nsegm);
}


// Fill a set of histograms for a give chamber 
void DTSegmentAnalysis::fillHistos(DTChamberId chamberId,
				   float posX,
				   float posY,
				   float phi,
				   float theta,
				   float chi2) {
  // FIXME: optimization of the number of searches
  if(histosPerCh.find(chamberId) == histosPerCh.end()) {
    bookHistos(chamberId);
  }
                                
  histosPerCh[chamberId][1]->Fill(posX);
  histosPerCh[chamberId][2]->Fill(posY);
  histosPerCh[chamberId][3]->Fill(posX, posY);
  histosPerCh[chamberId][4]->Fill(phi);
  histosPerCh[chamberId][5]->Fill(theta);
  histosPerCh[chamberId][6]->Fill(chi2);
}


