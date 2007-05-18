/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2007/03/22 18:52:02 $
 * $Revision: 1.18 $
 * \author M. Zanetti - INFN Padova
 *
 */

#include <DQM/DTMonitorModule/interface/DTDigiTask.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
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

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

DTDigiTask::DTDigiTask(const edm::ParameterSet& ps){
  
  debug = ps.getUntrackedParameter<bool>("debug", "false");
  if(debug)
    cout<<"[DTDigiTask]: Constructor"<<endl;

  outputFile = ps.getUntrackedParameter<string>("outputFile", "DTDigiSources.root");

  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  edm::Service<MonitorDaemon> daemon; 	 
  daemon.operator->();

  dbe->setVerbose(1);

}


DTDigiTask::~DTDigiTask(){

  if(debug)
    cout << "DTDigiTask: analyzed " << nevents << " events" << endl;

}

void DTDigiTask::endJob(){

  if(debug)
    cout<<"[DTDigiTask] endjob called!"<<endl;

  if ( (outputFile.size() != 0) && (parameters.getUntrackedParameter<bool>("writeHisto", true)) ) 
    dbe->save(outputFile);
  
  dbe->rmdir("DT/DTDigiTask");
}

void DTDigiTask::beginJob(const edm::EventSetup& context){

  if(debug)
    cout<<"[DTDigiTask]: BeginJob"<<endl;

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // tTrig 
  if (parameters.getUntrackedParameter<bool>("readDB", true)) 
    context.get<DTTtrigRcd>().get(tTrigMap);
  // t0s 
  if (parameters.getParameter<bool>("performPerWireT0Calibration")) 
    context.get<DTT0Rcd>().get(t0Map);
  // tMax (not yet from the DB)
  tMax = parameters.getParameter<int>("defaultTmax");

}





void DTDigiTask::bookHistos(const DTSuperLayerId& dtSL, string folder, string histoTag) {

  if (debug) cout<<"[DTDigiTask]: booking"<<endl;

  stringstream wheel; wheel << dtSL.wheel();	
  stringstream station; station << dtSL.station();	
  stringstream sector; sector << dtSL.sector();	
  stringstream superLayer; superLayer << dtSL.superlayer();

  dbe->setCurrentFolder("DT/DTDigiTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);

  if (debug) {
    cout<<"[DTDigiTask]: folder "<< "DT/DTDigiTask/Wheel" + wheel.str() +
      "/Station" + station.str() +
      "/Sector" + sector.str() + "/" + folder<<endl;
    cout<<"[DTDigiTask]: histoTag "<<histoTag<<endl;
  }
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str(); 

  if (debug) cout<<"[DTDigiTask]: histoName "<<histoName<<endl;

  if ( parameters.getUntrackedParameter<bool>("readDB", false) ) 
    tTrigMap->slTtrig( dtSL, tTrig, tTrigRMS); 
  else tTrig = parameters.getParameter<int>("defaultTtrig");
  

  if ( folder == "TimeBoxes") {
    string histoTitle = histoName + " (TDC Counts)";
    int timeBoxGranularity = parameters.getUntrackedParameter<int>("timeBoxGranularity",4);

    if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
      int maxTDCCounts = 6400 * parameters.getUntrackedParameter<int>("tdcRescale", 1);
      (digiHistos[histoTag])[dtSL.rawId()] = 
	dbe->book1D(histoName,histoTitle, maxTDCCounts/timeBoxGranularity, 0, maxTDCCounts);
    }    
    else {
      (digiHistos[histoTag])[dtSL.rawId()] = 
	dbe->book1D(histoName,histoTitle, 2*tMax/timeBoxGranularity, tTrig-tMax, tTrig+2*tMax);
    }
  }
  
  if ( folder == "CathodPhotoPeaks" ) 
    (digiHistos[histoTag])[dtSL.rawId()] = dbe->book1D(histoName,histoName,500,0,1000);
  
}


void DTDigiTask::bookHistos(const DTChamberId& dtCh, string folder, string histoTag) {

  if (debug) cout<<"[DTDigiTask]: booking"<<endl;
  
  stringstream wheel; wheel << dtCh.wheel();	
  stringstream station; station << dtCh.station();	
  stringstream sector; sector << dtCh.sector();	

  dbe->setCurrentFolder("DT/DTDigiTask/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str() + "/" + folder);

  if (debug){
    cout<<"[DTDigiTask]: folder "<< "DT/DTDigiTask/Wheel" + wheel.str() +
      "/Station" + station.str() +
      "/Sector" + sector.str() + "/" + folder<<endl;
    cout<<"[DTDigiTask]: histoTag "<<histoTag<<endl;
  }
  
  string histoName = histoTag 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 
  
  if (debug) cout<<"[DTDigiTask]: histoName "<<histoName<<endl;
  
  if ( folder == "Occupancies")    {
    
    const DTChamber* dtchamber = muonGeom->chamber(dtCh);
    const std::vector<const DTSuperLayer*> dtSupLylist = dtchamber->superLayers();
    std::vector<const DTSuperLayer*>::const_iterator suly = dtSupLylist.begin();
    std::vector<const DTSuperLayer*>::const_iterator sulyend = dtSupLylist.end();
    
    int nWires = 0;
    int nWires_max = 0;
    
    while(suly != sulyend) {
      const std::vector<const DTLayer*> dtLyList = (*suly)->layers();
      std::vector<const DTLayer*>::const_iterator ly = dtLyList.begin();
      std::vector<const DTLayer*>::const_iterator lyend = dtLyList.end();
      
      while(ly != lyend) {
	nWires = muonGeom->layer((*ly)->id())->specificTopology().channels();
	++ly;
	if(nWires > nWires_max) nWires_max = nWires;
	
      }
      ++suly;
    }
    
    (digiHistos[histoTag])[dtCh.rawId()] = dbe->book2D(histoName,histoName,nWires_max,1,nWires_max+1,12,0,12);
    
    for(int i=1;i<=12;i++) { 
      if(i<5){
	stringstream layer;
	string layer_name;
	layer<<i;
	layer>>layer_name;
	string label="SL1: L"+layer_name;
	(digiHistos[histoTag])[dtCh.rawId()]->setBinLabel(i,label,2);
      }
      else if(i>4 && i<9){
	stringstream layer;
	string layer_name;
	layer<<(i-4);
	layer>>layer_name;
	string label="SL2: L"+layer_name;
	(digiHistos[histoTag])[dtCh.rawId()]->setBinLabel(i,label,2);
      }
      else if(i>8 && i<13){
	stringstream layer;
	string layer_name;
	layer<<(i-8);
	layer>>layer_name;
	string label="SL3: L"+layer_name;
	(digiHistos[histoTag])[dtCh.rawId()]->setBinLabel(i,label,2);
      }
    }
    
  }
}


void DTDigiTask::analyze(const edm::Event& e, const edm::EventSetup& c){
  
  nevents++;
  if (nevents%1000 == 0 && debug) {}
  
  edm::Handle<DTDigiCollection> dtdigis;
  e.getByLabel("dtunpacker", dtdigis);
  
  if ( !parameters.getUntrackedParameter<bool>("localrun", true) ) e.getByType(ltcdigis);
  
  bool checkNoisyChannels = parameters.getUntrackedParameter<bool>("checkNoisyChannels","false");
  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels) {
    // Get the map of noisy channels
    c.get<DTStatusFlagRcd>().get(statusMap);
  }
  
  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){
      
      if(checkNoisyChannels) {
	const DTWireId wireId(((*dtLayerId_It).first), (*digiIt).wire());
	bool isNoisy = false;
	bool isFEMasked = false;
	bool isTDCMasked = false;
	bool isTrigMask = false;
	bool isDead = false;
	bool isNohv = false;
	statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
	if(isNoisy) {
	  continue;
	}      
      }
      
      // for clearness..
      const  DTSuperLayerId dtSLId = ((*dtLayerId_It).first).superlayerId();
      uint32_t indexSL = dtSLId.rawId();
      const  DTChamberId dtChId = dtSLId.chamberId(); 
      uint32_t indexCh = dtChId.rawId();
      int layer_number=((*dtLayerId_It).first).layer();
      int superlayer_number=dtSLId.superlayer();
      
      
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
      
      
      string histoTag;

      // TimeBoxes per SL
      histoTag = "TimeBox" + triggerSource();
      if (digiHistos[histoTag].find(indexSL) == digiHistos[histoTag].end())
	bookHistos( dtSLId, string("TimeBoxes"), histoTag );
      (digiHistos.find(histoTag)->second).find(indexSL)->second->Fill(tdcTime);

      
      // 2nd - 1st (CathodPhotoPeak) per SL
      if ( (*digiIt).number() == 1 ) {
	
	DTDigiCollection::const_iterator firstDigiIt = digiIt;
	firstDigiIt--;
	
	histoTag = "CathodPhotoPeak";
	if (digiHistos[histoTag].find(indexSL) == digiHistos[histoTag].end())
	  bookHistos( dtSLId, string("CathodPhotoPeaks"), histoTag );
	(digiHistos.find(histoTag)->second).find(indexSL)->second->Fill((*digiIt).countsTDC()-
									(*firstDigiIt).countsTDC());
      }


      // only for pre-Calibration jobs 
      if (parameters.getUntrackedParameter<bool>("preCalibrationJob", true)) {
	
	//Occupancies per chamber
	histoTag = "OccupancyAllHits_perCh";
	if (digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end())
	  bookHistos( dtChId, string("Occupancies"), histoTag );
	(digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*digiIt).wire(),
									(layer_number+(superlayer_number-1)*4)-1);
      }
      
      // after-Calibration jobs 
      else {
	
	// Noise: Before tTrig
	if (tdcTime < inTimeHitsLowerBound ) {
	  
	  //Occupancies Noise per chamber
	  histoTag = "OccupancyNoise_perCh";
	  if (digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end())
	    bookHistos( dtChId, string("Occupancies"), histoTag );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*digiIt).wire(),
									  (layer_number+(superlayer_number-1)*4)-1);
	}

	// Physical hits: within the time window	
	else if (tdcTime > inTimeHitsLowerBound && tdcTime < inTimeHitsUpperBound) { 

	  //Occupancies Signal per chamber
	  histoTag = "OccupancyInTimeHits_perCh";
	  if (digiHistos[histoTag].find(indexCh) == digiHistos[histoTag].end()) 
	    bookHistos( dtChId, string("Occupancies"), histoTag );
	  (digiHistos.find(histoTag)->second).find(indexCh)->second->Fill((*digiIt).wire(),
									  (layer_number+(superlayer_number-1)*4)-1);
	}
      }

    }
  }
  
}


string DTDigiTask::triggerSource() {

  string l1ASource;

  if ( !parameters.getUntrackedParameter<bool>("localrun", true) ){
    for (std::vector<LTCDigi>::const_iterator ltc_it = ltcdigis->begin(); ltc_it != ltcdigis->end(); ltc_it++){
      int otherTriggerSum=0;
      for (int i = 1; i < 6; i++)
	otherTriggerSum += int((*ltc_it).HasTriggered(i));
      
      if ((*ltc_it).HasTriggered(0) && otherTriggerSum == 0) 
	l1ASource = "DTonly";
      else if (!(*ltc_it).HasTriggered(0))
	l1ASource = "NoDT";
      else if ((*ltc_it).HasTriggered(0) && otherTriggerSum > 0)
	l1ASource = "DTalso";
    }
  }

  return l1ASource;

}

