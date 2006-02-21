/*
 * \file DTTestPulsesTask.cc
 * 
 * $Date: 2006/02/15 19:00:59 $
 * $Revision: 1.2 $
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
#include "Geometry/DTSimAlgo/interface/DTGeometry.h"
#include "Geometry/DTSimAlgo/interface/DTLayer.h"
#include "Geometry/CommonTopologies/interface/DTTopology.h"

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

  // Get the pedestals tTrig
  context.get<DTTtrigRcd>().get(tTrig_TPMap);

}


void DTTestPulsesTask::bookHistos(const DTLayerId& dtLayer, string folder, string histoTag) {


  stringstream wheel; wheel << dtLayer.wheel();	
  stringstream station; station << dtLayer.station();	
  stringstream sector; sector << dtLayer.sector();	
  stringstream superLayer; superLayer << dtLayer.superlayer();	
  stringstream layer; layer << dtLayer.layer();	

  cout<<"[DTTestPulseTask]: booking"<<endl;

  dbe->setCurrentFolder("DT/DTTestPulseTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);

  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str() 
    + "_L" + layer.str();

  // To be un-commented once the pedestal DB will work
//   if ( ! tTrigMap->slTtrig( dtLayer.wheel(),
// 			    dtLayer.station(),
// 			    dtLayer.sector(),
// 			    dtLayer.superlayer(), tTrig)) 
  tTrig_TP = parameters.getParameter<int>("defaultTtrig_TP");
  
  if (!parameters.getUntrackedParameter<bool>("t0sMeanFromDB",true))
    t0sPeakRange = make_pair( parameters.getUntrackedParameter<int>("t0sRangeLowerBound", -100) + tTrig_TP, 
			      parameters.getUntrackedParameter<int>("t0sRangeUpperBound", 100) + tTrig_TP);
  
  
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
      dbe->book2D(histoName,histoName,
		  nWires, 0, nWires, 
		  t0sPeakRange.first - t0sPeakRange.second, t0sPeakRange.first, t0sPeakRange.second);
  }
  else if ( folder == "TPTimeBox" ) {
    testPulsesTimeBoxes[int( DTLayerId(dtLayer.wheel(),
				       dtLayer.station(),
				       dtLayer.sector(),
				       dtLayer.superlayer(),
				       dtLayer.layer()).chamberId().rawId())] = 
      dbe->book1D(histoName,histoName, 2*tTrig_TP, 0, 2*tTrig_TP);
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
      int index = ((*dtLayerId_It).first).rawId();
      int chIndex = ((*dtLayerId_It).first).chamberId().rawId();

      if (testPulsesTimeBoxes.find(index) != testPulsesTimeBoxes.end())
	testPulsesHistos.find(index)->second->Fill(index,(*digiIt).countsTDC());
      else {
	bookHistos( (*dtLayerId_It).first , string("TPOccupancies"), string("TestPulses2D") );
	testPulsesHistos.find(index)->second->Fill(index,(*digiIt).countsTDC());
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

