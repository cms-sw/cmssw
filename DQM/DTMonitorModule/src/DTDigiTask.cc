/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2006/02/08 21:14:31 $
 * $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTDigiTask.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

// Digis
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTSimAlgo/interface/DTGeometry.h"
#include "Geometry/DTSimAlgo/interface/DTLayer.h"
#include "Geometry/CommonTopologies/interface/DTTopology.h"


#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>

using namespace edm;
using namespace std;

DTDigiTask::DTDigiTask(const edm::ParameterSet& ps){

  cout<<"[DTDigiTask]: Constructor"<<endl;

  outputFile = ps.getUntrackedParameter<string>("outputFile", "DTDigiSources.root");

  logFile.open("DTDigiTask.log");

  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  dbe->setVerbose(1);

  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();


  tMaxRescaled = ps.getParameter<int>("TMaxForInTimeHitDefinition");
  tTrigRescaling = ps.getParameter<int>("TTrigForInTimeHitDefinition");
  
}


DTDigiTask::~DTDigiTask(){

  cout << "DTDigiTask: analyzed " << nevents << " events" << endl;

  logFile.close();
  if ( outputFile.size() != 0 ) dbe->save(outputFile);
}


void DTDigiTask::beginJob(const edm::EventSetup& context){

  cout<<"[DTDigiTask]: BeginJob"<<endl;

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTDigiTask::bookHistos(const DTLayerId& dtLayer, string folder, string histoTag) {


  stringstream wheel; wheel << dtLayer.wheel();	
  stringstream station; station << dtLayer.station();	
  stringstream sector; sector << dtLayer.sector();	
  stringstream superLayer; superLayer << dtLayer.superlayer();	
  stringstream layer; layer << dtLayer.layer();	

  cout<<"[DTDigiTask]: booking"<<endl;

  dbe->setCurrentFolder("DT/DTDigiTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);

  cout<<"[DTDigiTask]: folder "<< "DT/DTDigiTask/Wheel" + wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" + folder<<endl;

  cout<<"[DTDigiTask]: histoTag "<<histoTag<<endl;

  string histoName = histoTag +"_SL" + superLayer.str() + "_L" + layer.str();

  cout<<"[DTDigiTask]: histoName "<<histoName<<endl;

  const int nWires = muonGeom->layer(DTLayerId(dtLayer.wheel(),
					       dtLayer.station(),
					       dtLayer.sector(),
					       dtLayer.superlayer(),
					       dtLayer.layer()))->specificTopology().channels();

  if ( folder == "Occupancies" ) {
    if ( histoTag == "HitsOrder" ) {
      (digiHistos[histoTag])[int(DTLayerId(dtLayer.wheel(),
					   dtLayer.station(),
					   dtLayer.sector(),
					   dtLayer.superlayer(),
					   dtLayer.layer()).rawId())] = 
	dbe->book1D(histoName,histoName,20,0,20);
    }
    else {
      (digiHistos[histoTag])[int(DTLayerId(dtLayer.wheel(),
					   dtLayer.station(),
					   dtLayer.sector(),
					   dtLayer.superlayer(),
					   dtLayer.layer()).rawId())] = 
	dbe->book1D(histoName,histoName,nWires,0,nWires);
    }
  }
  if ( folder == "TimeBoxes") {
    (digiHistos[histoTag])[int(DTLayerId(dtLayer.wheel(),
					 dtLayer.station(),
					 dtLayer.sector(),
					 dtLayer.superlayer(),
					 dtLayer.layer()).rawId())] = 
      dbe->book1D(histoName,histoName,1000,0,9000);
  }
  if ( folder == "CathodPhotoPeaks") {
    (digiHistos[histoTag])[int(DTLayerId(dtLayer.wheel(),
					 dtLayer.station(),
					 dtLayer.sector(),
					 dtLayer.superlayer(),
					 dtLayer.layer()).rawId())] = 
      dbe->book1D(histoName,histoName,500,0,1000);
  }
  
}


void DTDigiTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;
  if (nevents%100 == 0) 
    cout<<"[DTDigiTask]: "<<nevents<<" events analyzed"<<endl;

  int pippo;
  if (nevents == 4000) cin>>pippo;

  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);


  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){


      // for clearness..
      int index = ((*dtLayerId_It).first).rawId();

      // get the tTrig from the DB for the definition of InTimeHit
      // float tTrig = db.get("tTrig") - tTrigRescaling;
      float tTrig = tTrigRescaling;



      // Noise: Before tTrig
      if ((*digiIt).countsTDC() < tTrig ) {
	
	if (digiHistos[string("OccupancyNoise")].find(index) != 
	    digiHistos[string("OccupancyNoise")].end()) {
	  (digiHistos.find(string("OccupancyNoise"))->second).find(index)->second->Fill((*digiIt).wire());
	}
	else {
	  bookHistos( (*dtLayerId_It).first, string("Occupancies"), string("OccupancyNoise") );
	  (digiHistos.find(string("OccupancyNoise"))->second).find(index)->second->Fill((*digiIt).wire());
	  
	}
      }
      
      // time > tTrig
      else if ((*digiIt).countsTDC() > tTrig ) { 
	
// 	// Order of the after the tTrig
// 	if (digiHistos[string("HitsOrder")].find(index) != 
// 	    digiHistos[string("HitsOrder")].end()) {
// 	  (digiHistos.find(string("HitsOrder"))->second).find(index)->second->Fill((*digiIt).wire(),
// 										   (*digiIt).number());
// 	}
// 	else {
// 	  bookHistos( (*dtLayerId_It).first, string("Occupancies"), string("HitsOrder") );
// 	  (digiHistos.find(string("HitsOrder"))->second).find(index)->second->Fill((*digiIt).wire(),
// 										   (*digiIt).number());
	  
// 	}
	
	// Physical hits: into the time window	
	if ( (*digiIt).countsTDC() < tMaxRescaled + tTrig) {
	  
	  // Occupancies
	  if (digiHistos[string("OccupancyInTimeHits")].find(index) != 
	      digiHistos[string("OccupancyInTimeHits")].end()) {
	    (digiHistos.find(string("OccupancyInTimeHits"))->second).find(index)->second->Fill((*digiIt).wire());
	  }
	  else {
	    bookHistos( (*dtLayerId_It).first, string("Occupancies"), string("OccupancyInTimeHits") );
	    (digiHistos.find(string("OccupancyInTimeHits"))->second).find(index)->second->Fill((*digiIt).wire());
	  }
	  
	  // TimeBoxes
	  if (digiHistos[string("TimeBoxInTimeHits")].find(index) != 
	      digiHistos[string("TimeBoxInTimeHits")].end()) {
	    (digiHistos.find(string("TimeBoxInTimeHits"))->second).find(index)->second->Fill((*digiIt).countsTDC());
	  }
	  else {
	    bookHistos( (*dtLayerId_It).first, string("TimeBoxes"), string("TimeBoxInTimeHits") );
	    (digiHistos.find(string("TimeBoxInTimeHits"))->second).find(index)->second->Fill((*digiIt).countsTDC());
	    
	  }
	  
	  
//  	  // After pulses from a physical hit
// 	  if ( (*digiIt).number() > 0 ) {
	    
// 	    if (digiHistos[string("OccupancyAfterPulseHits")].find(index) != 
// 		digiHistos[string("OccupancyAfterPulseHits")].end()) {
// 	      (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
// 	    }
// 	    else {
// 	      bookHistos( (*dtLayerId_It).first, string("Occupancies"), string("OccupancyAfterPulseHits") );
// 	      (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
	      
// 	      (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
// 	    }
// 	  }
	  
// 	}
	
// 	// After pulses: after the time window
// 	if ( (*digiIt).countsTDC() > tMaxRescaled ) {
// 	  if (digiHistos[string("OccupancyAfterPulseHits")].find(index) != 
// 	      digiHistos[string("OccupancyAfterPulseHits")].end()) {
// 	    (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
// 	  }
// 	  else {
// 	    bookHistos( (*dtLayerId_It).first, string("Occupancies"), string("OccupancyAfterPulseHits") );
// 	    (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
// 	  }
// 	}
	
	}
      }
//       // 2nd - 1st
      if ( (*digiIt).number() == 1 ) {
	DTDigiCollection::const_iterator firstDigiIt = digiIt--;
	
 	if ( (*digiIt).countsTDC() > tMaxRescaled ) {
 	  if (digiHistos[string("CathodPhotoPeak")].find(index) != 
 	      digiHistos[string("CathodPhotoPeak")].end()) {
 	    (digiHistos.find(string("CathodPhotoPeak"))->second).find(index)->second->Fill((*digiIt).countsTDC()-
 											   (*firstDigiIt).countsTDC());
 	  }
 	  else {
 	    bookHistos( (*dtLayerId_It).first, string("CathodPhotoPeaks"), string("CathodPhotoPeak") );
 	    (digiHistos.find(string("CathodPhotoPeak"))->second).find(index)->second->Fill((*digiIt).countsTDC()-
 											   (*firstDigiIt).countsTDC());
 	  }
	}
	
      }
    }
  }
  
  
  
}

