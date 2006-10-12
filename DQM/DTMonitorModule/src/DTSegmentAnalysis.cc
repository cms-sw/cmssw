
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/10/09 18:19:41 $
 *  $Revision: 1.7 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTSegmentAnalysis.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "Geometry/Vector/interface/Pi.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include <iterator>

using namespace edm;
using namespace std;

DTSegmentAnalysis::DTSegmentAnalysis(const ParameterSet& pset,
				     DaqMonitorBEInterface* dbe) : theDbe(dbe) {
  debug = pset.getUntrackedParameter<bool>("debug","false");
  // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getParameter<string>("recHits4DLabel");
  parameters = pset;
  if(parameters.getUntrackedParameter<bool>("MTCC", false))
    {
      for(int wheel=1; wheel<3; wheel++)
	{
	  for(int sec=10; sec<10+wheel; sec++)
	    {
	      bookHistos(wheel,sec);
	      for (int st=1; st<5; st++)
		{
		  DTChamberId chId(wheel, st, sec);
		  bookHistos(chId);
		}
	    }
	  DTChamberId chId(wheel, 4, 14);
	  bookHistos(chId);
	  //bookHistos(wheel,14);
	}
    }
}

DTSegmentAnalysis::~DTSegmentAnalysis(){

}


void DTSegmentAnalysis::analyze(const Event& event, const EventSetup& setup) {
  if(debug)
    cout << "[DTSegmentAnalysis] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;
  
  //Get the trigger source from ltc digis
  DTTrig=-99, CSCTrig=-99, RBC1Trig=-99, RBC2Trig=-99, RPCTBTrig=-99;
  edm::Handle<LTCDigiCollection> ltcdigis;
  if ( !parameters.getUntrackedParameter<bool>("localrun", true) ) 
    {
      event.getByType(ltcdigis);
      for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
	if ((*ltc_it).HasTriggered(0))
	  DTTrig=1;
	if ((*ltc_it).HasTriggered(1))
	  CSCTrig=1;
	if ((*ltc_it).HasTriggered(2))
	  RBC1Trig=1;
	if ((*ltc_it).HasTriggered(3))
	  RBC2Trig=1;
	if ((*ltc_it).HasTriggered(4))
	  RPCTBTrig=1;
      }
    }

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
    if(debug)
      cout << "   Chamber: " << *chamberId << " has " << nsegm
	   << " 4D segments" << endl;
    fillHistos(*chamberId, nsegm);
    fillHistos(nsegm,(*chamberId).wheel(),(*chamberId).sector());
    // Loop over the rechits of this ChamerId
    for (DTRecSegment4DCollection::const_iterator segment4D = range.first;
	 segment4D!=range.second;
	   ++segment4D){
      LocalPoint segment4DLocalPos = (*segment4D).localPosition();
      LocalVector segment4DLocalDirection = (*segment4D).localDirection();
      int nHits = (((*segment4D).phiSegment())->specificRecHits()).size();
      if((*segment4D).hasZed()) 
	nHits = nHits + ((((*segment4D).zSegment())->specificRecHits()).size());
      
      if (segment4DLocalDirection.z()) {
	fillHistos(*chamberId,
		   nHits,
		   segment4DLocalPos.x(), 
		   segment4DLocalPos.y(),
		   atan(segment4DLocalDirection.x()/segment4DLocalDirection.z())* 180./Geom::pi(),
		   atan(segment4DLocalDirection.y()/segment4DLocalDirection.z())* 180./Geom::pi(),
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
			   "/Sector" + sector.str());
  // Create the monitor elements
  vector<MonitorElement *> histos;
  // Note hte order matters
  histos.push_back(theDbe->book1D("hN4DSeg"+chamberHistoName,
				  "# of 4D segments per event",
				  20, 0, 20));
  histos.push_back(theDbe->book1D("h4DSegmNHits"+chamberHistoName,
				  "# of hits per segment",
				  20, 0, 20));
  histos.push_back(theDbe->book1D("h4DSegmXInCham"+chamberHistoName,
				  "4D Segment X position (cm) in Chamer RF",
				  200, -200, 200));
  histos.push_back(theDbe->book1D("h4DSegmYInCham"+chamberHistoName,
				  "4D Segment Y position (cm) in Chamer RF",
				  200, -200, 200));
  histos.push_back(theDbe->book2D("h4DSegmXvsYInCham"+chamberHistoName,
				  "4D Segment position (cm) in Chamer RF",
				  200, -200, 200, 200, -200, 200));
  histos.push_back(theDbe->book1D("h4DSegmPhiDirection"+chamberHistoName,
				  "4D Segment Phi Direction (deg)",
				  180, -180, 180));
  histos.push_back(theDbe->book1D("h4DSegmThetaDirection"+chamberHistoName,
				  "4D Segment  Theta Direction (deg)",
				  180, -180, 180));
  histos.push_back(theDbe->book1D("h4DChi2"+chamberHistoName,
				  "4D Segment reduced Chi2",
				  30, 0, 30));
  histos.push_back(theDbe->book2D("h4DSegmThetaVSYInCham_DTTrig"+chamberHistoName,
				  "4D Segment  Theta(deg) VS Y (cm)",
				  125, -125, 125, 60, -60, 60));
  histosPerCh[chamberId] = histos;
}

void DTSegmentAnalysis::bookHistos(int w, int sec) {
  if(debug)
    cout << "   Booking histos for wheel " <<w<<"  sector "<<sec << endl;

  // Compose the chamber name
  stringstream wheel; wheel << w;	
  stringstream sect; sect << sec;
  
  string sectorHistoName =
    "_W" + wheel.str() +
    "_Sec" + sect.str();
  
  theDbe->setCurrentFolder("DT/DTLocalRecoTask/Wheel" + wheel.str());
  if (sec==14)
    sec=10;
  pair <int,int> sector;
  sector.first=w;
  sector.second=sec;
  histosPerSec[sector] = theDbe->book1D("hN4DSeg_Trigg"+sectorHistoName,"# of 4D segments per event per trigger source",50, 0, 50);
}

// Fill a set of histograms 
void DTSegmentAnalysis::fillHistos(int nsegm, int w, int sec) {
  if (sec==14)
    sec=10;
  pair <int,int> sector;
  sector.first=w;
  sector.second=sec;
  if(histosPerSec.find(sector) == histosPerSec.end()&& 
     !parameters.getUntrackedParameter<bool>("MTCC", false)) {
     bookHistos(w,sec);
  }
  if(DTTrig == 1)
    histosPerSec[sector]->Fill(nsegm);
  if(CSCTrig == 1)
    histosPerSec[sector]->Fill(nsegm+10);
  if(RBC1Trig == 1)
    histosPerSec[sector]->Fill(nsegm+20);
  if(RBC2Trig == 1)
    histosPerSec[sector]->Fill(nsegm+30);
  if(RPCTBTrig == 1)
    histosPerSec[sector]->Fill(nsegm+40);
}


// Fill a set of histograms for a give chamber 
void DTSegmentAnalysis::fillHistos(DTChamberId chamberId, int nsegm) {
  // FIXME: optimization of the number of searches
  if(histosPerCh.find(chamberId) == histosPerCh.end() && 
     !parameters.getUntrackedParameter<bool>("MTCC", false)) {
   bookHistos(chamberId);
  }
  histosPerCh[chamberId][0]->Fill(nsegm);
}


// Fill a set of histograms for a give chamber 
void DTSegmentAnalysis::fillHistos(DTChamberId chamberId,
				   int nHits,
				   float posX,
				   float posY,
				   float phi,
				   float theta,
				   float chi2) {
  // FIXME: optimization of the number of searches
  if((histosPerCh.find(chamberId) == histosPerCh.end()) && 
     !parameters.getUntrackedParameter<bool>("MTCC", false))  {
    bookHistos(chamberId);
  }
  vector<MonitorElement *> histos =  histosPerCh[chamberId];                          
  histos[1]->Fill(nHits);
  histos[2]->Fill(posX);
  histos[3]->Fill(posY);
  histos[4]->Fill(posX, posY);
  histos[5]->Fill(phi);
  histos[6]->Fill(theta);
  histos[7]->Fill(chi2);
  if (parameters.getUntrackedParameter<bool>("localrun", true) ) 
     histos[8]->Fill(posY,theta);
   else
     {
       if(DTTrig==1) histos[8]->Fill(posY,theta); 
     }
}
