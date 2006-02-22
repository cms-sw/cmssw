/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2006/02/21 19:04:14 $
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
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// T0s
#include <CondFormats/DTObjects/interface/DTT0.h>
#include <CondFormats/DataRecord/interface/DTT0Rcd.h>
#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>


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

  edm::Service<MonitorDaemon> daemon; 	 
  daemon.operator->();

  dbe->setVerbose(1);


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

  // Get the pedestals tTrig
  //context.get<DTTtrigRcd>().get(tTrigMap);
  if (parameters.getParameter<bool>("performPerWireT0Calibration")) context.get<DTT0Rcd>().get(t0Map);
 
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

  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str() 
    + "_L" + layer.str();

  cout<<"[DTDigiTask]: histoName "<<histoName<<endl;

  const int nWires = muonGeom->layer(DTLayerId(dtLayer.wheel(),
					       dtLayer.station(),
					       dtLayer.sector(),
					       dtLayer.superlayer(),
					       dtLayer.layer()))->specificTopology().channels();

  // To be un-commented once the pedestal DB will work
//   if ( ! tTrigMap->slTtrig( dtLayer.wheel(),
// 			    dtLayer.station(),
// 			    dtLayer.sector(),
// 			    dtLayer.superlayer(), tTrig)) 
    tTrig = parameters.getParameter<int>("defaultTtrig");
    
  tMax = parameters.getParameter<int>("defaultTmax");

  

  if ( folder == "Occupancies/Noise" ||
       folder == "Occupancies/Signal" ||
       folder == "Occupancies/AfterPulse" ) {
    (digiHistos[histoTag])[int(DTLayerId(dtLayer.wheel(),
					 dtLayer.station(),
					 dtLayer.sector(),
					 dtLayer.superlayer(),
					 dtLayer.layer()).rawId())] = 
      dbe->book1D(histoName,histoName,nWires,0,nWires);
  
  }
  if ( folder == "TimeBoxes") {
    (digiHistos[histoTag])[int(DTLayerId(dtLayer.wheel(),
					 dtLayer.station(),
					 dtLayer.sector(),
					 dtLayer.superlayer(),
					 dtLayer.layer()).rawId())] = 
      dbe->book1D(histoName,histoName, 
		  2*tMax/parameters.getUntrackedParameter<int>("timeBoxGranularity",1), tTrig-tMax, tTrig+2*tMax);
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

  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);


  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){


      // for clearness..
      int index = ((*dtLayerId_It).first).rawId();

  // To be un-commented once the pedestal DB will work
//       if ( ! tTrigMap->slTtrig( ((*dtLayerId_It).first).wheel(),
// 				((*dtLayerId_It).first).station(),
// 				((*dtLayerId_It).first).sector(),
// 				((*dtLayerId_It).first).superlayer(), tTrig)) 
	tTrig = parameters.getParameter<int>("defaultTtrig");
      
      int inTimeHitsLowerBound = tTrig - parameters.getParameter<int>("inTimeHitsLowerBound");
      int inTimeHitsUpperBound = tTrig + tMax + parameters.getParameter<int>("inTimeHitsUpperBound");

      int t0 = 0; float t0RMS = 0;
      int tdcTime = (*digiIt).countsTDC();
      if (parameters.getParameter<bool>("performPerWireT0Calibration"))
	if ( ! t0Map->cellT0(((*dtLayerId_It).first).wheel(),
			     ((*dtLayerId_It).first).station(),
			     ((*dtLayerId_It).first).sector(),
			     ((*dtLayerId_It).first).superlayer(),
			     ((*dtLayerId_It).first).layer(),
			     (*digiIt).wire(), t0, t0RMS) )
	   tdcTime += t0;

      // Noise: Before tTrig
      if (tdcTime < inTimeHitsLowerBound ) {
	
	if (digiHistos[string("OccupancyNoise")].find(index) != 
	    digiHistos[string("OccupancyNoise")].end()) {
	  (digiHistos.find(string("OccupancyNoise"))->second).find(index)->second->Fill((*digiIt).wire());
	} else {
	  bookHistos((*dtLayerId_It).first, string("Occupancies/Noise"), string("OccupancyNoise")); 
	  (digiHistos.find(string("OccupancyNoise"))->second).find(index)->second->Fill((*digiIt).wire());
	}
      }
      
      // time > tTrig
      else if (tdcTime > inTimeHitsLowerBound ) { 
	
	// Physical hits: into the time window	
	if ( tdcTime < inTimeHitsUpperBound) {
	  
	  // Occupancies
	  if (digiHistos[string("OccupancyInTimeHits")].find(index) != 
	      digiHistos[string("OccupancyInTimeHits")].end()) {
	    (digiHistos.find(string("OccupancyInTimeHits"))->second).find(index)->second->Fill((*digiIt).wire());
	  } else {
	    bookHistos( (*dtLayerId_It).first, string("Occupancies/Signal"), string("OccupancyInTimeHits") );
	    (digiHistos.find(string("OccupancyInTimeHits"))->second).find(index)->second->Fill((*digiIt).wire());
	  }
	  
	  // TimeBoxes
	  if (digiHistos[string("TimeBoxInTimeHits")].find(index) != 
	      digiHistos[string("TimeBoxInTimeHits")].end()) {
	    (digiHistos.find(string("TimeBoxInTimeHits"))->second).find(index)->second->Fill(tdcTime);
	  } else {
	    bookHistos( (*dtLayerId_It).first, string("TimeBoxes"), string("TimeBoxInTimeHits") );
	    (digiHistos.find(string("TimeBoxInTimeHits"))->second).find(index)->second->Fill(tdcTime);
	  }
	  
	  
  	  // After pulses from a physical hit
 	  if ( (*digiIt).number() > 0 ) {
	    if (digiHistos[string("OccupancyAfterPulseHits")].find(index) != 
 		digiHistos[string("OccupancyAfterPulseHits")].end()) {
 	      (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
 	    } else {
 	      bookHistos( (*dtLayerId_It).first, string("Occupancies/AfterPulse"), string("OccupancyAfterPulseHits") );
 	      (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
 	    }
 	  }
	  
 	}
	
 	// After pulses: after the time window
 	if ( tdcTime > inTimeHitsUpperBound ) {
 	  if (digiHistos[string("OccupancyAfterPulseHits")].find(index) != 
 	      digiHistos[string("OccupancyAfterPulseHits")].end()) {
 	    (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
 	  } else {
 	    bookHistos( (*dtLayerId_It).first, string("Occupancies/AfterPulse"), string("OccupancyAfterPulseHits") );
 	    (digiHistos.find(string("OccupancyAfterPulseHits"))->second).find(index)->second->Fill((*digiIt).wire());
 	  }
 	}
	
      }

       // 2nd - 1st
       if ( (*digiIt).number() == 1 ) {
	 
	 DTDigiCollection::const_iterator firstDigiIt = digiIt;
	 firstDigiIt--;

	 if (digiHistos[string("CathodPhotoPeak")].find(index) != 
	     digiHistos[string("CathodPhotoPeak")].end()) {
	   (digiHistos.find(string("CathodPhotoPeak"))->second).find(index)->second->Fill((*digiIt).countsTDC()-
											  (*firstDigiIt).countsTDC());
	 } else {
	   bookHistos( (*dtLayerId_It).first, string("CathodPhotoPeaks"), string("CathodPhotoPeak") );
	   (digiHistos.find(string("CathodPhotoPeak"))->second).find(index)->second->Fill((*digiIt).countsTDC()-
											  (*firstDigiIt).countsTDC());
	 }
       }
       
    }
  }
  
  
}

