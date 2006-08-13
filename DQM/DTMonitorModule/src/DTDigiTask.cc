/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2006/08/01 18:02:52 $
 * $Revision: 1.9 $
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
#include <math.h>

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

  // Get the pedestals 
  // tTrig 
  if ( !parameters.getUntrackedParameter<bool>("readDB", true)) 
    context.get<DTTtrigRcd>().get(tTrigMap);
  // t0s 
  if (parameters.getParameter<bool>("performPerWireT0Calibration")) 
    context.get<DTT0Rcd>().get(t0Map);


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

  if ( parameters.getUntrackedParameter<bool>("readDB", false) ) 
    tTrigMap->slTtrig( dtLayer.superlayerId(), tTrig, tTrigRMS); 
  else tTrig = parameters.getParameter<int>("defaultTtrig");
  
  tMax = parameters.getParameter<int>("defaultTmax");
  
  if ( folder == "Occupancies" ||
       folder == "Occupancies/Noise" ||
       folder == "Occupancies/Signal" ||
       folder == "Occupancies/AfterPulse" ) {
    (digiHistos[histoTag])[DTLayerId(dtLayer.wheel(),
				     dtLayer.station(),
				     dtLayer.sector(),
				     dtLayer.superlayer(),
				     dtLayer.layer()).rawId()] = 
      dbe->book1D(histoName,histoName,nWires,1,nWires+1);
  
  }

  if ( folder == "TimeBoxes") {
    string histoTitle = histoName + " (TDC Counts)";
    if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
      int maxTDCCounts = 6400 * parameters.getUntrackedParameter<int>("tdcRescale", 1);
      (digiHistos[histoTag])[DTLayerId(dtLayer.wheel(),
				       dtLayer.station(),
				       dtLayer.sector(),
				       dtLayer.superlayer(),
				       dtLayer.layer()).rawId()] = 
	dbe->book1D(histoName,histoTitle, maxTDCCounts/parameters.getUntrackedParameter<int>("timeBoxGranularity",4), 0, maxTDCCounts);
      
    }    
    else {
      (digiHistos[histoTag])[DTLayerId(dtLayer.wheel(),
				       dtLayer.station(),
				       dtLayer.sector(),
				       dtLayer.superlayer(),
				       dtLayer.layer()).rawId()] = 
	dbe->book1D(histoName,histoTitle, 
		    2*tMax/parameters.getUntrackedParameter<int>("timeBoxGranularity",4), tTrig-tMax, tTrig+2*tMax);
    }
  }

  if ( folder == "CathodPhotoPeaks" && !parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
    (digiHistos[histoTag])[DTLayerId(dtLayer.wheel(),
				     dtLayer.station(),
				     dtLayer.sector(),
				     dtLayer.superlayer(),
				     dtLayer.layer()).rawId()] = 
      dbe->book1D(histoName,histoName,500,0,1000);
  }

  /// FIXME: patch to provide tTrig to the Client. TO BE REMOVED once the ES will be accesible
  histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str();
  if ( folder == "tTrigRef" && !parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
    (digiHistos[histoTag])[DTSuperLayerId(dtLayer.wheel(),
					  dtLayer.station(),
					  dtLayer.sector(),
					  dtLayer.superlayer()).rawId()] = 
      dbe->book1D(histoName,histoName,10000,0,10000);
  }

}


void DTDigiTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;
  if (nevents%1000 == 0) 
    cout<<"[DTDigiTask]: "<<nevents<<" events analyzed"<<endl;

  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);


  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){


      // for clearness..
      uint32_t index = ((*dtLayerId_It).first).rawId();
      uint32_t indexSL = ((*dtLayerId_It).first).superlayerId().rawId();

      if ( parameters.getUntrackedParameter<bool>("readDB", false) ) 
	tTrigMap->slTtrig( ((*dtLayerId_It).first).superlayerId(), tTrig, tTrigRMS); 
      else tTrig = parameters.getParameter<int>("defaultTtrig");
      
      int inTimeHitsLowerBound = int(round(tTrig)) - parameters.getParameter<int>("inTimeHitsLowerBound");
      int inTimeHitsUpperBound = int(round(tTrig)) + tMax + parameters.getParameter<int>("inTimeHitsUpperBound");

      float t0; float t0RMS;
      int tdcTime = (*digiIt).countsTDC();

      if (parameters.getParameter<bool>("performPerWireT0Calibration")) {
	const DTWireId dtWireId(((*dtLayerId_It).first), (*digiIt).wire());
	t0Map->cellT0(dtWireId, t0, t0RMS) ;
	tdcTime += int(round(t0));
      }


      /* * * * * * * * * * * * * * */
      /* S T A R T   F I L L I N G */
      /* * * * * * * * * * * * * * */
      
      // only for pre-Calibration jobs 
      if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
	// Occupancies
	if (digiHistos[string("OccupancyAllHits")].find(index) != 
	    digiHistos[string("OccupancyAllHits")].end()) {
	  (digiHistos.find(string("OccupancyAllHits"))->second).find(index)->second->Fill((*digiIt).wire());
	} else {
	  bookHistos( (*dtLayerId_It).first, string("Occupancies"), string("OccupancyAllHits") );
	  (digiHistos.find(string("OccupancyAllHits"))->second).find(index)->second->Fill((*digiIt).wire());
	}
	
	// TimeBoxes
	if (digiHistos[string("TimeBoxAllHits")].find(index) != 
	    digiHistos[string("TimeBoxAllHits")].end()) {
	  (digiHistos.find(string("TimeBoxAllHits"))->second).find(index)->second->Fill(tdcTime);
	} else {
	  bookHistos( (*dtLayerId_It).first, string("TimeBoxes"), string("TimeBoxAllHits") );
	  (digiHistos.find(string("TimeBoxAllHits"))->second).find(index)->second->Fill(tdcTime);
	}
      }

      // after-Calibration jobs 
      else {

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
       

	// Fake tTrig histo
	if (digiHistos[string("tTrigRef")].find(indexSL) != 
	      digiHistos[string("tTrigRef")].end()) {
	    (digiHistos.find(string("tTrigRef"))->second).find(indexSL)->second->Fill(tTrig);
	} else {
	  bookHistos( (*dtLayerId_It).first, string("tTrigRef"), string("tTrigRef") );
	  cout<<"[DTDigiTask]::analyze()-DEBUG 1"<<endl;
	  (digiHistos.find(string("tTrigRef"))->second).find(indexSL)->second->Fill(tTrig);
	  cout<<"[DTDigiTask]::analyze()-DEBUG 2"<<endl;
	} 

      }
    }
  }
  
  
}

