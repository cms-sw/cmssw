 /*
 * \file DTDigiForNoiseTask.cc
 * 
 * $Date: 2011/11/12 09:18:41 $
 * $Revision: 1.11 $
 * \author G. Mila - INFN Torino
 *
 */

#include <DQM/DTMonitorModule/src/DTDigiForNoiseTask.h>

// Framework
#include <FWCore/Framework/interface/EventSetup.h>

// Digis
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;


DTDigiForNoiseTask::DTDigiForNoiseTask(const edm::ParameterSet& ps){
  
  debug = ps.getUntrackedParameter<bool>("debug", false);
  if(debug)
    cout<<"[DTDigiForNoiseTask]: Constructor"<<endl;

  parameters = ps;
  
  dbe = edm::Service<DQMStore>().operator->();

}


DTDigiForNoiseTask::~DTDigiForNoiseTask(){

  if(debug)
    cout << "DTDigiForNoiseTask: analyzed " << nevents << " events" << endl;

}


void DTDigiForNoiseTask::endJob(){

  if(debug)
    cout<<"[DTDigiForNoiseTask] endjob called!"<<endl;

  dbe->rmdir("DT/DTDigiForNoiseTask");

}


void DTDigiForNoiseTask::beginJob(){

  if(debug)
    cout<<"[DTDigiForNoiseTask]: BeginJob"<<endl;

  nevents = 0;

}

void DTDigiForNoiseTask::beginRun(const edm::Run& run, const edm::EventSetup& context) {

  if(debug)
    cout<<"[DTDigiForNoiseTask]: BeginRun"<<endl;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}

void DTDigiForNoiseTask::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  if(debug)
    cout<<"[DTDigiForNoiseTask]: Begin of LS transition"<<endl;
  
  if(lumiSeg.id().luminosityBlock()%parameters.getUntrackedParameter<int>("ResetCycle", 3) == 0) {
    for(map< DTLayerId, MonitorElement* > ::const_iterator histo = digiHistos.begin();
	histo != digiHistos.end();
	histo++) {
      (*histo).second->Reset();
    }
  }
  
}


void DTDigiForNoiseTask::bookHistos(const DTLayerId& lId) {
  
  if (debug) cout<<"[DTDigiForNoiseTask]: booking"<<endl;

  const  DTSuperLayerId dtSLId = lId.superlayerId();
  const  DTChamberId dtChId = dtSLId.chamberId(); 
  stringstream layer; layer << lId.layer();
  stringstream superLayer; superLayer << dtSLId.superlayer();
  stringstream wheel; wheel << dtChId.wheel();	
  stringstream station; station << dtChId.station();	
  stringstream sector; sector << dtChId.sector();

  dbe->setCurrentFolder("DT/DTDigiForNoiseTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/DigiPerEvent");

  if (debug){
    cout<<"[DTDigiForNoiseTask]: folder "<< "DT/DTDigiTask/Wheel" + wheel.str() +
      "/Station" + station.str() +
      "/Sector" + sector.str() + "/DigiPerEvent"<<endl;
  }
  
  string histoName =
    "DigiPerEvent_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str()
    + "_SL" + superLayer.str()  
    + "_L" + layer.str();
  
  if (debug) cout<<"[DTDigiTask]: histoName "<<histoName<<endl;

  const DTTopology& dtTopo = muonGeom->layer(lId)->specificTopology();
  const int firstWire = dtTopo.firstChannel();
  const int lastWire = dtTopo.lastChannel();
  int nWires = lastWire-firstWire+1;
  
  digiHistos[lId] = dbe->book2D(histoName,histoName,nWires,firstWire,lastWire,10,-0.5,9.5);

}
  

void DTDigiForNoiseTask::analyze(const edm::Event& e, const edm::EventSetup& c){
  
  nevents++;
  //  cout << "events:  " << nevents << endl;
  if (nevents%1000 == 0 && debug) {}
  
  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);

  std::map< int,int > DigiPerWirePerEvent;
  
  // Loop over all the chambers
  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();
  // Loop over the SLs
  for (; ch_it != ch_end; ++ch_it) {
    //    DTChamberId ch = (*ch_it)->id();
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();
    // Loop over the SLs
    for(; sl_it != sl_end; ++sl_it) {
      vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin(); 
      vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
      // Loop over the Ls
      for(; l_it != l_end; ++l_it) {
	DTLayerId layerId = (*l_it)->id();

	DTDigiCollection::Range layerDigi= dtdigis->get(layerId);
	if(layerDigi.first != layerDigi.second){

	  const DTTopology& dtTopo = muonGeom->layer(layerId)->specificTopology();
	  const int firstWire = dtTopo.firstChannel();
	  const int lastWire = dtTopo.lastChannel();
	  
	  if (digiHistos.find(layerId) == digiHistos.end())
	    bookHistos(layerId);
	  
	  if (digiHistos.find(layerId) != digiHistos.end()){
	    for (int wire=firstWire; wire-lastWire <= 0; wire++) {
	      DigiPerWirePerEvent[wire]= 0;
	    }

	    for (DTDigiCollection::const_iterator digi = layerDigi.first;
		 digi!=layerDigi.second;
		 ++digi){
	      DigiPerWirePerEvent[(*digi).wire()]+=1;
	    }

	    for (int wire=firstWire; wire-lastWire<=0; wire++) {
	      digiHistos.find(layerId)->second->Fill(wire,DigiPerWirePerEvent[wire]);
	    }
	  }
	}

      } //Loop Ls
    } //Loop SLs
  } //Loop over chambers
   
}
