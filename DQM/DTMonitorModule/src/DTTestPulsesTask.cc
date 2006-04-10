/*
 * \file DTTestPulsesTask.cc
 * 
 * $Date: 2006/03/24 16:18:29 $
 * $Revision: 1.5 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTTestPulsesTask.h>

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
#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

using namespace edm;
using namespace std;


DTTestPulsesTask::DTTestPulsesTask(const edm::ParameterSet& ps){


  cout<<"[DTTestPulseTask]: Constructor"<<endl;

  outputFile = ps.getUntrackedParameter<string>("outputFile", "DTTestPulesSources.root");

  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  edm::Service<MonitorDaemon> daemon; 	 
  daemon.operator->();

  dbe->setVerbose(1);

}

DTTestPulsesTask::~DTTestPulsesTask(){

  cout <<"[DTTestPulsesTask]: analyzed " << nevents << " events" << endl;
  
  if ( outputFile.size() != 0 ) dbe->save(outputFile);
}


void DTTestPulsesTask::beginJob(const edm::EventSetup& context){

  cout<<"[DTTestPulsesTask]: BeginJob"<<endl;

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // Get the pedestals tTrig (always get it, even if the tTrig_TP is taken from conf)
  context.get<DTTtrigRcd>().get(tTrig_TPMap);

}


void DTTestPulsesTask::bookHistos(const DTLayerId& dtLayer, string folder, string histoTag) {


  stringstream wheel; wheel << dtLayer.wheel();	
  stringstream station; station << dtLayer.station();	
  stringstream sector; sector << dtLayer.sector();	
  stringstream superLayer; superLayer << dtLayer.superlayer();	
  stringstream layer; layer << dtLayer.layer();	

  cout<<"[DTTestPulseTask]: booking"<<endl;

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


  if ( parameters.getUntrackedParameter<bool>("readDB", false) ) {
    if ( ! tTrig_TPMap->slTtrig( dtLayer.wheel(),
				 dtLayer.station(),
				 dtLayer.sector(),
				 dtLayer.superlayer(), tTrig_TP)) 
      tTrig_TP = parameters.getParameter<int>("defaultTtrig_TP");
  }
  else tTrig_TP = parameters.getParameter<int>("defaultTtrig_TP");
  

  // keep the Range around the tTrig in order to keep track of it in the histos
  t0sPeakRange = make_pair( parameters.getUntrackedParameter<int>("t0sRangeLowerBound", -100) + tTrig_TP, 
			    parameters.getUntrackedParameter<int>("t0sRangeUpperBound", 100) + tTrig_TP);
  

  cout<<"t0sRangeLowerBound "<<t0sPeakRange.first<<"; "
      <<"t0sRangeUpperBound "<<t0sPeakRange.second<<endl;
  
  if ( folder == "TPOccupancies" ) {

    const int nWires = muonGeom->layer(DTLayerId(dtLayer.wheel(),
						 dtLayer.station(),
						 dtLayer.sector(),
						 dtLayer.superlayer(),
						 dtLayer.layer()))->specificTopology().channels();
    
    testPulsesHistos[int(DTLayerId(dtLayer.wheel(),
				   dtLayer.station(),
				   dtLayer.sector(),
				   dtLayer.superlayer(),
				   dtLayer.layer()).rawId())] =
      dbe->bookProfile(histoName,histoName,
		       nWires, 0, nWires, // Xaxis: channels
		       t0sPeakRange.first - t0sPeakRange.second, t0sPeakRange.first, t0sPeakRange.second); // Yaxis: times
  }
  else if ( folder == "TPTimeBox" ) {
    // Time Box per Chamber
    testPulsesTimeBoxes[int( DTLayerId(dtLayer.wheel(),
				       dtLayer.station(),
				       dtLayer.sector(),
				       dtLayer.superlayer(),
				       dtLayer.layer()).chamberId().rawId())] = 
      dbe->book1D(histoName,histoName, 10000, 0, 10000); // Overview of the TP (and noise) times
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

      if (testPulsesHistos.find(layerIndex) != testPulsesHistos.end())
	testPulsesHistos.find(layerIndex)->second->Fill((*digiIt).channel(),(*digiIt).countsTDC());
      else {
	bookHistos( (*dtLayerId_It).first , string("TPOccupancies"), string("TestPulses2D") );
	testPulsesHistos.find(layerIndex)->second->Fill((*digiIt).channel(),(*digiIt).countsTDC());
      }
	
      if (testPulsesTimeBoxes.find(chIndex) != testPulsesTimeBoxes.end())
	testPulsesTimeBoxes.find(chIndex)->second->Fill((*digiIt).countsTDC());
      else {
	bookHistos( (*dtLayerId_It).first , string("TPTimeBox"), string("TestPulsesTB") );
	testPulsesTimeBoxes.find(chIndex)->second->Fill((*digiIt).countsTDC());
      }
    }
  }

}

