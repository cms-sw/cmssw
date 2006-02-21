/*
 * \file DTDigiClient.cc
 *
 * $Date: 2006/01/18 11:40:54 $
 * $Revision: 1.79 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorClient/interface/DTDigiClient.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "Geometry/DTSimAlgo/interface/DTGeometry.h"
#include "Geometry/DTSimAlgo/interface/DTLayer.h"

// ROOT Staff
#include "TROOT.h"
#include "TStyle.h"
#include "TGaxis.h"

using namespace edm;
using namespace std;


DTDigiClient::DTDigiClient(const edm::ParameterSet& ps){

  parameters = ps;

  numOfEvents = 0;

  // resetting counters 
  updates = 0;
  last_operation = 0;

  // DQM default client name
  string clientName = ps.getUntrackedParameter<string>("clientName", "DTDigiClient");

  // DQM default collector host name
  string hostName = ps.getUntrackedParameter<string>("hostName", "localhost");

  // DQM default host port
  int hostPort = ps.getUntrackedParameter<int>("hostPort", 9090);

  cout << " Client '" << clientName << "' " << endl
       << " Collector on host '" << hostName << "'"
       << " on port '" << hostPort << "'" << endl;

  // DQM ROOT output
  outputFile = ps.getUntrackedParameter<string>("outputFile", "DTDigiResults.root");

  // start DQM user interface instance
  dtMUI = new MonitorUIRoot(hostName, hostPort, clientName);

  // subscribe to DTDigi Occupancies folder
  dtMUI->subscribe("*/DT/DTDigiTask/Wheel*/Station*/Sector*/Occupancies/*");

  // ROOT sytle
  setROOTStyle();

}

DTDigiClient::~DTDigiClient(){

  if ( outputFile.size() != 0 ) dtMUI->save(outputFile);

  sleep(10);

  dtMUI->unsubscribe("*/DT/DTDigiTask/Wheel*/Station*/Sector*/Occupancies/*");
  dtMUI->disconnect();
  if ( dtMUI ) delete dtMUI;

}


void DTDigiClient::beginJob(const edm::EventSetup& context) {

  context.get<DTTtrigRcd>().get(tTrigMap);
  
}

bool DTDigiClient::receiveMonitoring() {

  bool go_on = dtMUI->update();
  
  dtMUI->subscribeNew("*/DT/DTDigiTask/Wheel*/Station*/Sector*/Occupancies/*");
  
  updates = dtMUI->getNumUpdates();
  
  return go_on;
}



void DTDigiClient::analyze(const edm::Event& e, const edm::EventSetup& context){

  numOfEvents++;
  if ( numOfEvents % 1000 == 0 ) 
    LogInfo("DTDigiClientGeneral")<<"[DTDigiClient]"<<numOfEvents<<"events analyzed";
  

  if ( receiveMonitoring() ) {

    // loop over chambers
    for (int w = -2; w <= 2; ++w) {
      stringstream wheel; wheel << w;
      for (int st = 1; st <= 4; ++st) {
	stringstream station; station << st;
	for (int sec = 1; sec <= 12; ++sec) {
	  stringstream sector; sector << sec;


	  // clear the bad channels
	  badChannels.clear();

	  noiseAnalysis(DTChamberId(w,st,sec), context);

	  inTimeHitsAnalysis(DTChamberId(w,st,sec));

	  afterPulsesAnalysis(DTChamberId(w,st,sec));

	  timeBoxAnalysis();
	}
      }
    }

  }

}


void DTDigiClient::noiseAnalysis(DTChamberId dtChId, const edm::EventSetup& context) {

  if( updates % parameters.getUntrackedParameter<int>("noiseAnalysisPeriodiciy", 100) == 0 &&
      updates != last_operation) {

    stringstream wheel; wheel << dtChId.wheel();
    stringstream station; station << dtChId.station();
    stringstream sector; sector << dtChId.sector();


    // loop on superlayer
    for (int sl = 1; sl <= 3; ++sl) {
      stringstream superLayer; superLayer << sl;

      /* ****************************************************************
	 Get the tTrig from the DB in order to get the normalization.
	 Here the number of events is assumed to be the number of updates
         **************************************************************** */ 
      int tTrig;
      if ( ! tTrigMap->slTtrig( dtChId.wheel(),
				dtChId.station(),
				dtChId.sector(),
				sl, tTrig)) 
	tTrig = parameters.getParameter<int>("defaultTtrig");
      const float ns_s = 1e9*(32/25);
      float norm = ns_s/float(tTrig*updates);
      
      // loop on layer
      for (int l = 1; l <= 4; ++l) {
	stringstream layer; layer << l;
	
	// get the noise histogram  
	string folderName = 
	  "DT/DTDigiTask/Wheel" + wheel.str() +
	  "/Station" + station.str() +
	  "/Sector" + sector.str() + "/Occupancies/Noise/";
	string histoName = "OccupancyNoise_W" + wheel.str()
	  + "_St" + station.str()
	  + "_Sec" + sector.str()
	  + "_SL" + superLayer.str() 
	  + "_L" + layer.str();

	MonitorElement * noise = dtMUI->get(folderName+histoName);
	if(noise) {
	  cout<<"Pippo"<<endl;
	  
	  
	}		
      }
    }

    checkNoise();

    last_operation = updates;
  }
  
}




void DTDigiClient::checkNoise() {


}


void DTDigiClient::inTimeHitsAnalysis(DTChamberId dtChId) {

}

void DTDigiClient::afterPulsesAnalysis(DTChamberId dtChId) {

}

void DTDigiClient::timeBoxAnalysis() {


}


void DTDigiClient::setROOTStyle() {

  gStyle->Reset("Default");
  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetFillColor(10);
  gStyle->SetStatColor(10);
  gStyle->SetTitleColor(10);
  gStyle->SetTitleFillColor(10);
  TGaxis::SetMaxDigits(4);
  gStyle->SetOptTitle(kTRUE);
  gStyle->SetTitleX(0.00);
  gStyle->SetTitleY(1.00);
  gStyle->SetTitleW(0.00);
  gStyle->SetTitleH(0.06);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleFont(43, "c");
  gStyle->SetTitleFontSize(11);
  gStyle->SetOptStat(kFALSE);
  gStyle->SetStatX(0.99);
  gStyle->SetStatY(0.99);
  gStyle->SetStatW(0.25);
  gStyle->SetStatH(0.20);
  gStyle->SetStatBorderSize(1);
  gStyle->SetStatFont(43);
  gStyle->SetStatFontSize(10);
  gStyle->SetOptFit(kFALSE);
  gROOT->ForceStyle();

}
