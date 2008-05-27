
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/05/06 13:26:47 $
 *  $Revision: 1.9 $
 *  \author G. Cerminara - INFN Torino
 *  revised by G. Mila - INFN Torino
 */

#include "DTSegmentAnalysisTask.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include <iterator>

using namespace edm;
using namespace std;

DTSegmentAnalysisTask::DTSegmentAnalysisTask(const edm::ParameterSet& pset) {

  debug = pset.getUntrackedParameter<bool>("debug","false");
  if(debug)
    cout << "[DTSegmentAnalysisTask] Constructor called!" << endl;

  detailedAnalysis = pset.getUntrackedParameter<bool>("detailedAnalysis","false");

  // Get the DQM needed services
  theDbe = edm::Service<DQMStore>().operator->();
  theDbe->setCurrentFolder("DT/DTSegments");

  parameters = pset;

}


DTSegmentAnalysisTask::~DTSegmentAnalysisTask(){
  if(debug)
    cout << "[DTSegmentAnalysisTask] Destructor called!" << endl;
}


void DTSegmentAnalysisTask::beginJob(const edm::EventSetup& context){
 
  // the name of the 4D rec hits collection
  theRecHits4DLabel = parameters.getParameter<string>("recHits4DLabel");

   // Get the DT Geometry
  context.get<MuonGeometryRecord>().get(dtGeom);

  // loop over all the DT chambers & book the histos
  vector<DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = dtGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    bookHistos((*ch_it)->id());
  }

}


void DTSegmentAnalysisTask::endJob(){
 
  if(debug)
    cout<<"[DTSegmentAnalysisTask] endjob called!"<<endl;

  theDbe->rmdir("DT/DTSegments");
}
  


void DTSegmentAnalysisTask::analyze(const edm::Event& event, const edm::EventSetup& setup) {

  if(debug)
    cout << "[DTSegmentAnalysisTask] Analyze #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;
  if(!(event.id().event()%1000) && debug)
    {
      cout << "[DTSegmentAnalysisTask] Analyze #Run: " << event.id().run()
	   << " #Event: " << event.id().event() << endl;
    }


  // Get the map of noisy channels
  bool checkNoisyChannels = parameters.getUntrackedParameter<bool>("checkNoisyChannels","false");
  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    setup.get<DTStatusFlagRcd>().get(statusMap);
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

    // Loop over the rechits of this ChamerId
    for (DTRecSegment4DCollection::const_iterator segment4D = range.first;
	 segment4D!=range.second;
	   ++segment4D){

      //FOR NOISY CHANNELS////////////////////////////////
     bool segmNoisy = false;
     if((*segment4D).hasPhi()){
       const DTChamberRecSegment2D* phiSeg = (*segment4D).phiSegment();
       vector<DTRecHit1D> phiHits = phiSeg->specificRecHits();
       map<DTSuperLayerId,vector<DTRecHit1D> > hitsBySLMap; 
       for(vector<DTRecHit1D>::const_iterator hit = phiHits.begin();
	   hit != phiHits.end(); ++hit) {
	 DTWireId wireId = (*hit).wireId();
	 
	 // Check for noisy channels to skip them
	 if(checkNoisyChannels) {
	   bool isNoisy = false;
	   bool isFEMasked = false;
	   bool isTDCMasked = false;
	   bool isTrigMask = false;
	   bool isDead = false;
	   bool isNohv = false;
	   statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	   if(isNoisy) {
	     if(debug)
	       cout << "Wire: " << wireId << " is noisy, skipping!" << endl;
	     segmNoisy = true;
	  }      
	 }
       }
     }

     if((*segment4D).hasZed()) {
       const DTSLRecSegment2D* zSeg = (*segment4D).zSegment();  // zSeg lives in the SL RF
       // Check for noisy channels to skip them
       vector<DTRecHit1D> zHits = zSeg->specificRecHits();
       for(vector<DTRecHit1D>::const_iterator hit = zHits.begin();
	   hit != zHits.end(); ++hit) {
	 DTWireId wireId = (*hit).wireId();
	 if(checkNoisyChannels) {
	   bool isNoisy = false;
	    bool isFEMasked = false;
	    bool isTDCMasked = false;
	    bool isTrigMask = false;
	    bool isDead = false;
	    bool isNohv = false;
	    //cout<<"wire id "<<wireId<<endl;
	    statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	    if(isNoisy) {
	      if(debug)
		cout << "Wire: " << wireId << " is noisy, skipping!" << endl;
	      segmNoisy = true;
	    }      
	 }
       }
     } 
     
     if (segmNoisy) {
       if(debug)
	 cout<<"skipping the segment: it contains noisy cells"<<endl;
       continue;
     }
     //END FOR NOISY CHANNELS////////////////////////////////
      
     int nHits=0;
     LocalPoint segment4DLocalPos = (*segment4D).localPosition();
     LocalVector segment4DLocalDirection = (*segment4D).localDirection();
     if((*segment4D).hasPhi())
       nHits = (((*segment4D).phiSegment())->specificRecHits()).size();
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
	if(debug)
	  cout << "[DTSegmentAnalysisTask] Warning: segment local direction is: "
	       << segment4DLocalDirection << endl;
      }
    }
  }

  // -----------------------------------------------------------------------------
}
  

// Book a set of histograms for a give chamber
void DTSegmentAnalysisTask::bookHistos(DTChamberId chamberId) {
  if(debug)
    cout << "   Booking histos for chamber: " << chamberId << endl;

  // Compose the chamber name
  stringstream wheel; wheel << chamberId.wheel();	
  stringstream station; station << chamberId.station();	
  stringstream sector; sector << chamberId.sector();	
  
  string chamberHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();
  
  theDbe->setCurrentFolder("DT/DTSegments/Wheel" + wheel.str() +
			   "/Station" + station.str() +
			   "/Sector" + sector.str());
  // Create the monitor elements
  vector<MonitorElement *> histos;

  histos.push_back(theDbe->book1D("hN4DSeg"+chamberHistoName,
				  "# of 4D segments per event",
				  20, 0, 20));
  histos.push_back(theDbe->book1D("h4DSegmNHits"+chamberHistoName,
				  "# of hits per segment",
				  20, 0, 20));
  if(detailedAnalysis){
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
  }
  histosPerCh[chamberId] = histos;
}


// Fill a set of histograms for a given chamber 
void DTSegmentAnalysisTask::fillHistos(DTChamberId chamberId, int nsegm) {
  histosPerCh[chamberId][0]->Fill(nsegm);
}

// Fill a set of histograms for a give chamber 
void DTSegmentAnalysisTask::fillHistos(DTChamberId chamberId,
				   int nHits,
				   float posX,
				   float posY,
				   float phi,
				   float theta,
				   float chi2) {

  vector<MonitorElement *> histos =  histosPerCh[chamberId];                          
  histos[1]->Fill(nHits);
  if(detailedAnalysis){
    histos[2]->Fill(posX);
    histos[3]->Fill(posY);
    histos[4]->Fill(posX, posY);
    histos[5]->Fill(phi);
    histos[6]->Fill(theta);
    histos[7]->Fill(chi2);
  }

}
