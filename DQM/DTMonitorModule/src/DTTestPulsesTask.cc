/*
 * \file DTTestPulsesTask.cc
 * 
 * $Date: 2010/01/05 10:14:40 $
 * $Revision: 1.16 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTTestPulsesTask.h>

// Framework
#include <FWCore/Framework/interface/EventSetup.h>

// Digis
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// Pedestals
#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>
#include <CondFormats/DTObjects/interface/DTRangeT0.h>
#include <CondFormats/DataRecord/interface/DTRangeT0Rcd.h>
#include "DQMServices/Core/interface/DQMStore.h"


using namespace edm;
using namespace std;


DTTestPulsesTask::DTTestPulsesTask(const edm::ParameterSet& ps){


  cout<<"[DTTestPulseTask]: Constructor"<<endl;

  parameters = ps;


  t0sPeakRange = make_pair( parameters.getUntrackedParameter<int>("t0sRangeLowerBound", -100), 
			    parameters.getUntrackedParameter<int>("t0sRangeUpperBound", 100));

  
  dbe = edm::Service<DQMStore>().operator->();

}

DTTestPulsesTask::~DTTestPulsesTask(){

  cout <<"[DTTestPulsesTask]: analyzed " << nevents << " events" << endl;
 
}


void DTTestPulsesTask::beginJob(){

  cout<<"[DTTestPulsesTask]: BeginJob"<<endl;

  nevents = 0;

}

void DTTestPulsesTask::beginRun(const edm::Run& run, const edm::EventSetup& context) {

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // Get the pedestals tTrig (always get it, even if the TPRange is taken from conf)
  //context.get<DTRangeT0Rcd>().get(t0RangeMap);

}

void DTTestPulsesTask::bookHistos(const DTLayerId& dtLayer, string folder, string histoTag) {


  stringstream wheel; wheel << dtLayer.wheel();	
  stringstream station; station << dtLayer.station();	
  stringstream sector; sector << dtLayer.sector();	
  stringstream superLayer; superLayer << dtLayer.superlayer();	
  stringstream layer; layer << dtLayer.layer();	

  cout<<"[DTTestPulseTask]: booking"<<endl;

  // TP Profiles  
  if ( folder == "TPProfile" ) {

    const int nWires = muonGeom->layer(DTLayerId(dtLayer.wheel(),
						 dtLayer.station(),
						 dtLayer.sector(),
						 dtLayer.superlayer(),
						 dtLayer.layer()))->specificTopology().channels();

    dbe->setCurrentFolder("DT/DTTestPulsesTask/Wheel" + wheel.str() +
			  "/Station" + station.str() +
			  "/Sector" + sector.str() + 
			  "/SuperLayer" + superLayer.str() + 
			  "/" +folder);
    
    string histoName = histoTag 
      + "_W" + wheel.str() 
      + "_St" + station.str() 
      + "_Sec" + sector.str() 
      + "_SL" + superLayer.str() 
      + "_L" + layer.str();

    // Setting the range 
    if ( parameters.getUntrackedParameter<bool>("readDB", false) ) {
      t0RangeMap->slRangeT0( dtLayer.superlayerId() , t0sPeakRange.first, t0sPeakRange.second);
    }
    
    
    cout<<"t0sRangeLowerBound "<<t0sPeakRange.first<<"; "
	<<"t0sRangeUpperBound "<<t0sPeakRange.second<<endl;
    
    
    testPulsesProfiles[int(DTLayerId(dtLayer.wheel(),
				   dtLayer.station(),
				   dtLayer.sector(),
				   dtLayer.superlayer(),
				   dtLayer.layer()).rawId())] =
      dbe->bookProfile(histoName,histoName,
		       nWires, 0, nWires, // Xaxis: channels
		       t0sPeakRange.first - t0sPeakRange.second, t0sPeakRange.first, t0sPeakRange.second); // Yaxis: times
  }

  // TP Occupancies
  else if ( folder == "TPOccupancy" ) {

    dbe->setCurrentFolder("DT/DTTestPulsesTask/Wheel" + wheel.str() +
			  "/Station" + station.str() +
			  "/Sector" + sector.str() + 
			  "/SuperLayer" + superLayer.str() + 
			  "/" +folder);
    
    string histoName = histoTag 
      + "_W" + wheel.str() 
      + "_St" + station.str() 
      + "_Sec" + sector.str() 
      + "_SL" + superLayer.str() 
      + "_L" + layer.str();
    
    const int nWires = muonGeom->layer(DTLayerId(dtLayer.wheel(),
						 dtLayer.station(),
						 dtLayer.sector(),
						 dtLayer.superlayer(),
						 dtLayer.layer()))->specificTopology().channels();
    
    testPulsesOccupancies[int(DTLayerId(dtLayer.wheel(),
					dtLayer.station(),
					dtLayer.sector(),
					dtLayer.superlayer(),
					dtLayer.layer()).rawId())] =
      dbe->book1D(histoName, histoName, nWires, 0, nWires); 
  }

  // Time Box per Chamber
  else if ( folder == "TPTimeBox" ) {

    dbe->setCurrentFolder("DT/DTTestPulsesTask/Wheel" + wheel.str() +
			  "/Station" + station.str() +
			  "/Sector" + sector.str() + 
			  "/" +folder);
    
    string histoName = histoTag 
      + "_W" + wheel.str() 
      + "_St" + station.str() 
      + "_Sec" + sector.str(); 
    
    testPulsesTimeBoxes[int( DTLayerId(dtLayer.wheel(),
				       dtLayer.station(),
				       dtLayer.sector(),
				       dtLayer.superlayer(),
				       dtLayer.layer()).chamberId().rawId())] = 
      dbe->book1D(histoName, histoName, 10000, 0, 10000); // Overview of the TP (and noise) times
  }

}


void DTTestPulsesTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  nevents++;
  
  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);
  
  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){
      
      // for clearness..
      int layerIndex = ((*dtLayerId_It).first).rawId();
      int chIndex = ((*dtLayerId_It).first).chamberId().rawId();


      if ((int)(*digiIt).countsTDC() > t0sPeakRange.first &&
	  (int)(*digiIt).countsTDC() < t0sPeakRange.second ) {

	// Occupancies
	if (testPulsesOccupancies.find(layerIndex) != testPulsesOccupancies.end())
	  testPulsesOccupancies.find(layerIndex)->second->Fill((*digiIt).wire());
	else {
	  bookHistos( (*dtLayerId_It).first , string("TPOccupancy"), string("TestPulses") );
	  testPulsesOccupancies.find(layerIndex)->second->Fill((*digiIt).wire());
	}

	// Profiles
	if (testPulsesProfiles.find(layerIndex) != testPulsesProfiles.end())
	  testPulsesProfiles.find(layerIndex)->second->Fill((*digiIt).wire(),(*digiIt).countsTDC());
	else {
	  bookHistos( (*dtLayerId_It).first , string("TPProfile"), string("TestPulses2D") );
	  testPulsesProfiles.find(layerIndex)->second->Fill((*digiIt).wire(),(*digiIt).countsTDC());
	}
      }

      // Time Box
      if (testPulsesTimeBoxes.find(chIndex) != testPulsesTimeBoxes.end())
	testPulsesTimeBoxes.find(chIndex)->second->Fill((*digiIt).countsTDC());
      else {
	bookHistos( (*dtLayerId_It).first , string("TPTimeBox"), string("TestPulsesTB") );
	testPulsesTimeBoxes.find(chIndex)->second->Fill((*digiIt).countsTDC());
      }
    }
  }

}

