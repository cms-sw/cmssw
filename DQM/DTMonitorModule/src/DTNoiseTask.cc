
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/06/30 12:50:03 $
 *  $Revision: 1.1 $
 *  \authors G. Mila , G. Cerminara - INFN Torino
 */

#include "DTNoiseTask.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/ServiceRegistry/interface/Service.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// Digi
#include <DataFormats/DTDigi/interface/DTDigi.h>
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

// RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

// Database
#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include <sstream>
#include <string>

using namespace edm;
using namespace std;



DTNoiseTask::DTNoiseTask(const ParameterSet& ps) : evtNumber(0) {

  cout<< "[DTNoiseTask]: Constructor"<<endl;

  dbe = edm::Service<DQMStore>().operator->();

  //switch for timeBox booking
  doTimeBoxHistos = ps.getUntrackedParameter<bool>("doTbHistos", false);
  
  // the name of the 4D rec hits collection
  theRecHits4DLabel = ps.getParameter<string>("recHits4DLabel");
  
}




DTNoiseTask::~DTNoiseTask(){}



/// BeginJob
void DTNoiseTask::beginJob(const edm::EventSetup& setup) {

  cout<< "[DTNoiseTask]: BeginJob"<<endl;

  // tTrig Map
  edm::ESHandle<DTTtrig> tTrigMap;
  setup.get<DTTtrigRcd>().get(tTrigMap);

  // get the geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Loop over all the chambers 	 
  vector<DTChamber*>::const_iterator ch_it = dtGeom->chambers().begin(); 	 
  vector<DTChamber*>::const_iterator ch_end = dtGeom->chambers().end(); 	 
  for (; ch_it != ch_end; ++ch_it) { 	 
    DTChamberId chId = (*ch_it)->id();
    // histo booking
    bookHistos(chId);
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 	 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end(); 	 
    // Loop over the SLs 	 
    for(; sl_it != sl_end; ++sl_it) { 
      DTSuperLayerId slId = (*sl_it)->id();
      if(doTimeBoxHistos)
	bookHistos(slId);
      float tTrig, tTrigRMS;
      tTrigMap->slTtrig(slId, tTrig, tTrigRMS);
      // tTrig mapping per station
      if(tTrigStMap.find(chId)==tTrigStMap.end() || 
	 (tTrigStMap.find(chId)!=tTrigStMap.end() && tTrig<tTrigStMap[chId]))
	tTrigStMap[chId]=int(tTrig);
    }
  }

}



/// To reset the MEs
void DTNoiseTask::beginLuminosityBlock(const edm::LuminosityBlock&  lumiSeg,
				       const edm::EventSetup& context) {

  cout<<"[DTNoiseTask]: Begin of LS transition"<<endl;

  for(map<DTChamberId, MonitorElement* > ::const_iterator histo = noiseHistos.begin();
	histo != noiseHistos.end();
	histo++) {
    (*histo).second->Reset();
  }

  if(doTimeBoxHistos){
    for(map<DTSuperLayerId, MonitorElement* > ::const_iterator histo = tbHistos.begin();
	histo != tbHistos.end();
	histo++) {
      (*histo).second->Reset();
    }
  }

}


  
/// Analyze
void DTNoiseTask::analyze(const edm::Event& e, const edm::EventSetup& c) {

  evtNumber++;
  if(evtNumber%1000==0)
    cout<<"[DTNoiseTask]: Analyzing evt number :"<<evtNumber<<endl;

  // map of the chambers with at least 1 segment
  std::map<DTChamberId, int> segmentsChId;

  // Get the 4D segment collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  e.getByLabel(theRecHits4DLabel, all4DSegments);
  
  // Loop over all chambers containing a segment and look for the number of segments
  DTRecSegment4DCollection::id_iterator chamberId;
  for (chamberId = all4DSegments->id_begin();
       chamberId != all4DSegments->id_end();
       ++chamberId){
    segmentsChId[*chamberId]=1;
  }
 
  // Get the digis from the event
  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);
  
  // LOOP OVER ALL THE DIGIS OF THE EVENT
  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){
  
      //Check the TDC trigger width
      int tdcTime = (*digiIt).countsTDC();
      int upperLimit = tTrigStMap[(*dtLayerId_It).first.superlayerId().chamberId()]-200;
      if(doTimeBoxHistos)
	tbHistos[(*dtLayerId_It).first.superlayerId()]->Fill(tdcTime);
      if(tdcTime>upperLimit)
	continue;

      //Check the chamber has no 4D segments
      if(segmentsChId.find((*dtLayerId_It).first.superlayerId().chamberId())!=segmentsChId.end())
	continue;

      // fill the occupancy histo
      TH2F* noise_root = noiseHistos[(*dtLayerId_It).first.superlayerId().chamberId()]->getTH2F();
      double normalization=0;
      if(mapEvt.find((*dtLayerId_It).first.superlayerId().chamberId())!=mapEvt.end()){
	normalization = upperLimit*mapEvt[(*dtLayerId_It).first.superlayerId().chamberId()];
	noise_root->Scale(normalization);
      }
      int yBin=(*dtLayerId_It).first.layer()+(4*((*dtLayerId_It).first.superlayerId().superlayer()-1));
      noise_root->Fill((*digiIt).wire(),yBin);
      // normalize the occupancy histo
      mapEvt[(*dtLayerId_It).first.superlayerId().chamberId()]=evtNumber;
      normalization = 1/double(upperLimit*mapEvt[(*dtLayerId_It).first.superlayerId().chamberId()]);
      noise_root->Scale(normalization);

    }
  }

  //if(evtNumber%1000==0)
  //dbe->save("prova.root");

  
}
  


/// Endjob
void DTNoiseTask::endJob() {}


void DTNoiseTask::bookHistos(DTChamberId chId) {

  // set the folder
  stringstream wheel; wheel << chId.wheel();	
  stringstream station; station << chId.station();	
  stringstream sector; sector << chId.sector();	
  dbe->setCurrentFolder("DT/05-Noise/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str());

  // Build the histo name
  string histoName = string("NoiseRate")
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() ;
  
  cout<< "[DTNoiseTask]: booking chamber histo:"<<endl;
  cout<< "              folder "<< "DT/01-Noise/Wheel" + wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/"<<endl; 
  cout<< "              histoName "<<histoName<<endl;

  // Get the chamber from the geometry
  int nWires_max = 0;
  const DTChamber* dtchamber = dtGeom->chamber(chId);
  const vector<const DTSuperLayer*> superlayers = dtchamber->superLayers();

  // Loop over layers and find the max # of wires
  for(vector<const DTSuperLayer*>::const_iterator sl = superlayers.begin();
      sl != superlayers.end(); ++sl) { // loop over SLs
    vector<const DTLayer*> layers = (*sl)->layers();
    for(vector<const DTLayer*>::const_iterator lay = layers.begin();
	lay != layers.end(); ++lay) { // loop over layers
      int nWires = (*lay)->specificTopology().channels();
      if(nWires > nWires_max) nWires_max = nWires;
    }
  }

  noiseHistos[chId] = dbe->book2D(histoName,"Noise rate (Hz) per channel", nWires_max,1, nWires_max+1,12,1,13);
  noiseHistos[chId]->setAxisTitle("wire number",1);
  noiseHistos[chId]->setBinLabel(1,"SL1_L1",2);
  noiseHistos[chId]->setBinLabel(2,"SL1_L2",2);
  noiseHistos[chId]->setBinLabel(3,"SL1_L3",2);
  noiseHistos[chId]->setBinLabel(4,"SL1_L4",2);
  noiseHistos[chId]->setBinLabel(5,"SL2_L1",2);
  noiseHistos[chId]->setBinLabel(6,"SL2_L2",2);
  noiseHistos[chId]->setBinLabel(7,"SL2_L3",2);
  noiseHistos[chId]->setBinLabel(8,"SL2_L4",2);
  noiseHistos[chId]->setBinLabel(9,"SL3_L1",2);
  noiseHistos[chId]->setBinLabel(10,"SL3_L2",2);
  noiseHistos[chId]->setBinLabel(11,"SL3_L3",2);
  noiseHistos[chId]->setBinLabel(12,"SL3_L4",2);

}


void DTNoiseTask::bookHistos(DTSuperLayerId slId) {

  // set the folder
  stringstream wheel; wheel << slId.chamberId().wheel();	
  stringstream station; station << slId.chamberId().station();	
  stringstream sector; sector << slId.chamberId().sector();	
  stringstream superlayer; superlayer << slId.superlayer();
  dbe->setCurrentFolder("DT/05-Noise/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str());

  // Build the histo name
  string histoName = string("TimeBox")
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str()
    + "_SL" + superlayer.str();
  
  cout<<"[DTNoiseTask]: booking SL histo:"<<endl;
  cout<<"              folder "<< "DT/01-Noise/Wheel" + wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" << endl; 
  cout<<"              histoName "<<histoName<<endl;

  tbHistos[slId] = dbe->book1D(histoName,"Time Box (TDC counts)", 1000, 0, 6000);

}
