
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/06/27 09:43:36 $
 *  $Revision: 1.5 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/interface/DTSegmentsTask.h"

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
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include <iterator>

using namespace edm;
using namespace std;

DTSegmentsTask::DTSegmentsTask(const edm::ParameterSet& pset) {

  debug = pset.getUntrackedParameter<bool>("debug",false);

  // Get the DQM needed services
  theDbe = edm::Service<DQMStore>().operator->();
  theDbe->setVerbose(1);

  parameters = pset;

}


DTSegmentsTask::~DTSegmentsTask(){
}


void DTSegmentsTask::beginJob(void){
 
 // the name of the 4D rec hits collection
  theRecHits4DLabel = parameters.getParameter<string>("recHits4DLabel");

  theDbe->setCurrentFolder("Muons/DTSegmentsMonitor");

  // histos for phi segments
  phiHistos.push_back(theDbe->book2D("phiSegments_numHitsVsWheel", "phiSegments_numHitsVsWheel", 5, -2.5, 2.5, 20, 0, 20));
  phiHistos[0]->setBinLabel(1,"W-2",1);
  phiHistos[0]->setBinLabel(2,"W-1",1);
  phiHistos[0]->setBinLabel(3,"W0",1);
  phiHistos[0]->setBinLabel(4,"W1",1);
  phiHistos[0]->setBinLabel(5,"W2",1);
  phiHistos.push_back(theDbe->book2D("phiSegments_numHitsVsSector", "phiSegments_numHitsVsSector", 14, 0.5, 14.5, 20, 0, 20));
  phiHistos[1]->setBinLabel(1,"Sec1",1);
  phiHistos[1]->setBinLabel(2,"Sec2",1);
  phiHistos[1]->setBinLabel(3,"Sec3",1);
  phiHistos[1]->setBinLabel(4,"Sec4",1);
  phiHistos[1]->setBinLabel(5,"Sec5",1);
  phiHistos[1]->setBinLabel(6,"Sec6",1);
  phiHistos[1]->setBinLabel(7,"Sec7",1);
  phiHistos[1]->setBinLabel(8,"Sec8",1);
  phiHistos[1]->setBinLabel(9,"Sec9",1);
  phiHistos[1]->setBinLabel(10,"Sec10",1);
  phiHistos[1]->setBinLabel(11,"Sec11",1);
  phiHistos[1]->setBinLabel(12,"Sec12",1);
  phiHistos[1]->setBinLabel(13,"Sec13",1);
  phiHistos[1]->setBinLabel(14,"Sec14",1);
  phiHistos.push_back(theDbe->book2D("phiSegments_numHitsVsStation", "phiSegments_numHitsVsStation", 4, 0.5, 4.5, 20, 0, 20));
  phiHistos[2]->setBinLabel(1,"St1",1);
  phiHistos[2]->setBinLabel(2,"St2",1);
  phiHistos[2]->setBinLabel(3,"St3",1);
  phiHistos[2]->setBinLabel(4,"St4",1);

  // histos for theta segments
  thetaHistos.push_back(theDbe->book2D("thetaSegments_numHitsVsWheel", "thetaSegments_numHitsVsWheel", 5, -2.5, 2.5, 20, 0, 20));
  thetaHistos[0]->setBinLabel(1,"W-2",1);
  thetaHistos[0]->setBinLabel(2,"W-1",1);
  thetaHistos[0]->setBinLabel(3,"W0",1);
  thetaHistos[0]->setBinLabel(4,"W1",1);
  thetaHistos[0]->setBinLabel(5,"W2",1);
  thetaHistos.push_back(theDbe->book2D("thetaSegments_numHitsVsSector", "thetaSegments_numHitsVsSector", 14, 0.5, 14.5, 20, 0, 20));
  thetaHistos[1]->setBinLabel(1,"Sec1",1);
  thetaHistos[1]->setBinLabel(2,"Sec2",1);
  thetaHistos[1]->setBinLabel(3,"Sec3",1);
  thetaHistos[1]->setBinLabel(4,"Sec4",1);
  thetaHistos[1]->setBinLabel(5,"Sec5",1);
  thetaHistos[1]->setBinLabel(6,"Sec6",1);
  thetaHistos[1]->setBinLabel(7,"Sec7",1);
  thetaHistos[1]->setBinLabel(8,"Sec8",1);
  thetaHistos[1]->setBinLabel(9,"Sec9",1);
  thetaHistos[1]->setBinLabel(10,"Sec10",1);
  thetaHistos[1]->setBinLabel(11,"Sec11",1);
  thetaHistos[1]->setBinLabel(12,"Sec12",1);
  thetaHistos[1]->setBinLabel(13,"Sec13",1);
  thetaHistos[1]->setBinLabel(14,"Sec14",1);
  thetaHistos.push_back(theDbe->book2D("thetaSegments_numHitsVsStation", "thetaSegments_numHitsVsStation", 4, 0.5, 4.5, 20, 0, 20));
  thetaHistos[2]->setBinLabel(1,"St1",1);
  thetaHistos[2]->setBinLabel(2,"St2",1);
  thetaHistos[2]->setBinLabel(3,"St3",1);
  thetaHistos[2]->setBinLabel(4,"St4",1);

}


void DTSegmentsTask::endJob(){
  bool outputMEsInRootFile = parameters.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    theDbe->save(outputFileName);
  }

  theDbe->rmdir("DT/DTSegmentsTask");
}
  
void DTSegmentsTask::analyze(const edm::Event& event, const edm::EventSetup& setup) {

//  if(!(event.id().event()%1000) && debug)
//    {
//      cout << "[DTSegmentsTask] Analyze #Run: " << event.id().run()
//	   << " #Event: " << event.id().event() << endl;
//    }


  // Get the map of noisy channels
  bool checkNoisyChannels = parameters.getUntrackedParameter<bool>("checkNoisyChannels",false);
  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    setup.get<DTStatusFlagRcd>().get(statusMap);
  } 

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

     // Fill the histos
     int nHits=0;
     if((*segment4D).hasPhi()){
       nHits = (((*segment4D).phiSegment())->specificRecHits()).size();
       if(debug)
	 cout<<"Phi segment with number of hits: "<<nHits<<endl;
       phiHistos[0]->Fill((*chamberId).wheel(), nHits);
       phiHistos[1]->Fill((*chamberId).sector(), nHits);
       phiHistos[2]->Fill((*chamberId).station(), nHits);
     }
     if((*segment4D).hasZed()) {
       nHits = (((*segment4D).zSegment())->specificRecHits()).size();
       if(debug)
	 cout<<"Zed segment with number of hits: "<<nHits<<endl;
       thetaHistos[0]->Fill((*chamberId).wheel(), nHits);
       thetaHistos[1]->Fill((*chamberId).sector(), nHits);
       thetaHistos[2]->Fill((*chamberId).station(), nHits);
     }

    } //loop over segments
  } // loop over chambers

}

