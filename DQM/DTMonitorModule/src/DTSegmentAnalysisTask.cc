
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/11/06 17:36:20 $
 *  $Revision: 1.4 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTSegmentAnalysisTask.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

//Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"

//Digis & RecHit
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
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

  // Get the DQM needed services
  theDbe = edm::Service<DaqMonitorBEInterface>().operator->();
  theDbe->setVerbose(1);
  edm::Service<MonitorDaemon>().operator->();
  theDbe->setCurrentFolder("DT/DTSegmentAnalysisTask");

  parameters = pset;

}


DTSegmentAnalysisTask::~DTSegmentAnalysisTask(){
  if(debug)
    cout << "[DTSegmentAnalysisTask] Destructor called!" << endl;
}


void DTSegmentAnalysisTask::beginJob(const edm::EventSetup& context){
  // the name of the 4D rec hits collection
  theRecHits4DLabel = parameters.getParameter<string>("recHits4DLabel");
}


void DTSegmentAnalysisTask::endJob(){
 if(debug)
    cout<<"[DTSegmentAnalysisTask] endjob called!"<<endl;

  theDbe->rmdir("DT/DTSegmentAnalysisTask");
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
  int nsegm_W1Sec10=0, nsegm_W2Sec10=0, nsegm_W2Sec11=0;
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
    if((*chamberId).wheel()==1 &&((*chamberId).sector()==10 ||(*chamberId).sector()==14))
      nsegm_W1Sec10 = nsegm_W1Sec10+nsegm;
    if((*chamberId).wheel()==2 &&((*chamberId).sector()==10||(*chamberId).sector()==14))
      nsegm_W2Sec10 = nsegm_W2Sec10+nsegm;
    if((*chamberId).wheel()==2 &&((*chamberId).sector()==11))
      nsegm_W2Sec11 = nsegm_W2Sec11+nsegm;
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
   fillHistos(nsegm_W1Sec10,1,10);
   fillHistos(nsegm_W2Sec10,2,10);
   fillHistos(nsegm_W2Sec11,2,11);
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
  //   stringstream superLayer; superLayer << chamberId.superlayer();	
  //   stringstream layer; layer << chamberId.layer();	
  
  string chamberHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();
  
  theDbe->setCurrentFolder("DT/DTSegmentAnalysisTask/Wheel" + wheel.str() +
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

void DTSegmentAnalysisTask::bookHistos(int w, int sec) {
  if(debug)
    cout << "   Booking histos for wheel " <<w<<"  sector "<<sec << endl;

  // Compose the chamber name
  stringstream wheel; wheel << w;	
  stringstream sect; sect << sec;
  
  string sectorHistoName =
    "_W" + wheel.str() +
    "_Sec" + sect.str();
  
  theDbe->setCurrentFolder("DT/DTSegmentAnalysisTask/Wheel" + wheel.str());
  if (sec==14)
    sec=10;
  pair <int,int> sector;
  sector.first=w;
  sector.second=sec;
  histosPerSec[sector] = theDbe->book1D("hN4DSeg_Trigg"+sectorHistoName,"# of 4D segments per event per trigger source",500, 0, 500);
  histosPerSec[sector]->setBinLabel(0,"DTTrig",1);
  histosPerSec[sector]->setBinLabel(100,"CSCTrig",1);
  histosPerSec[sector]->setBinLabel(200,"RBC1Trig",1);
  histosPerSec[sector]->setBinLabel(300,"RBC2Trig",1);
  histosPerSec[sector]->setBinLabel(400,"RPCTBTrig",1);

}

// Fill a set of histograms 
void DTSegmentAnalysisTask::fillHistos(int nsegm, int w, int sec) {
  if (sec==14)
    sec=10;
  pair <int,int> sector;
  sector.first=w;
  sector.second=sec;
  if(histosPerSec.find(sector) == histosPerSec.end()) {
    bookHistos(w,sec);
  }
  if(DTTrig == 1)
    histosPerSec[sector]->Fill(nsegm);
  if(CSCTrig == 1)
    histosPerSec[sector]->Fill(nsegm+100);
  if(RBC1Trig == 1)
    histosPerSec[sector]->Fill(nsegm+200);
  if(RBC2Trig == 1)
    histosPerSec[sector]->Fill(nsegm+300);
  if(RPCTBTrig == 1)
    histosPerSec[sector]->Fill(nsegm+400);
}


// Fill a set of histograms for a give chamber 
void DTSegmentAnalysisTask::fillHistos(DTChamberId chamberId, int nsegm) {
  // FIXME: optimization of the number of searches
  if(histosPerCh.find(chamberId) == histosPerCh.end()) {
   bookHistos(chamberId);
  }
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
  // FIXME: optimization of the number of searches
  if(histosPerCh.find(chamberId) == histosPerCh.end())  {
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
