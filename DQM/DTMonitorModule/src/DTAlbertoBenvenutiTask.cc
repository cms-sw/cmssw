

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/06/19 10:20:52 $
 *  $Revision: 1.10 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorModule/src/DTAlbertoBenvenutiTask.h>

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
#include "TH1F.h"
#include "TFile.h"
#include "TPostScript.h"
#include "TCanvas.h"

using namespace edm;
using namespace std;


DTAlbertoBenvenutiTask::DTAlbertoBenvenutiTask(const edm::ParameterSet& ps){
  
  debug = ps.getUntrackedParameter<bool>("debug", "false");
  if(debug)
    cout<<"[DTAlbertoBenvenutiTask]: Constructor"<<endl;

  outputFile = ps.getUntrackedParameter<string>("outputFile", "DTDigiSources.root");
  maxTDCHits = ps.getUntrackedParameter<int>("maxTDCHits",1000);
  
  parameters = ps; 

}


DTAlbertoBenvenutiTask::~DTAlbertoBenvenutiTask(){

  if(debug)
    cout << "DTAlbertoBenvenutiTask: analyzed " << nevents << " events" << endl;

}


void DTAlbertoBenvenutiTask::endJob(){

  if(debug)
    cout<<"[DTAlbertoBenvenutiTask] endjob called!"<<endl;

  map< DTChamberId, vector<TH1F*> > TBMap_perChamber;

  for(map<DTWireId, TH1F* >::const_iterator wHisto = TBMap.begin();
      wHisto != TBMap.end();
      wHisto++) {
    DTChamberId chId = (*wHisto).first.layerId().superlayerId().chamberId();
    TBMap_perChamber[chId].push_back((*wHisto).second);
  }
 
  
  for(map<DTChamberId, vector<TH1F*> >::const_iterator Histo = TBMap_perChamber.begin();
      Histo != TBMap_perChamber.end();
      Histo++) {
    stringstream station; station << (*Histo).first.station();
    stringstream sector; sector << (*Histo).first.sector();	
    stringstream wheel; wheel << (*Histo).first.wheel();
    
    string fileTag = "TimeBoxes";
    string fileName = fileTag
      + "_W" + wheel.str()
      + "_Sec" + sector.str()
      + "_St" + station.str() 
      + ".ps";

    TPostScript psFile(fileName.c_str(),111);
    psFile.Range(20,26);
    int counter = 0;
    TCanvas c1("c1","",600,780);
    c1.Divide(2,3);
    psFile.NewPage();
    
    cout<<"[DTAlbertoBenvenutiTask] filling the file: "<<fileName<<endl;
    for(vector<TH1F*>::const_iterator tbHisto = (*Histo).second.begin();
	tbHisto != (*Histo).second.end();
	tbHisto++) {
      counter++;
      c1.cd(counter);
      (*tbHisto)->Draw();      
       if(counter%6 == 0 && counter>=6){
	 c1.Update();
	 psFile.NewPage();
	 c1.Clear();
	 c1.Divide(2,3);
	 counter=0;
       }
    } // loop over TB histos (divided per chamber)

  } //loop over the chambers

}


void DTAlbertoBenvenutiTask::beginJob(const edm::EventSetup& context){

  if(debug)
    cout<<"[DTAlbertoBenvenutiTask]: BeginJob"<<endl;

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


void DTAlbertoBenvenutiTask::bookHistos(const DTWireId dtWire) {

  if (debug) cout<<"[DTAlbertoBenvenutiTask]: booking"<<endl;

  stringstream wheel; wheel << dtWire.layerId().superlayerId().chamberId().wheel();	
  stringstream station; station << dtWire.layerId().superlayerId().chamberId().station();	
  stringstream sector; sector << dtWire.layerId().superlayerId().chamberId().sector();	
  stringstream superLayer; superLayer << dtWire.layerId().superlayerId().superlayer();
  stringstream layer; layer << dtWire.layerId().layer();
  stringstream wire; wire << dtWire.wire();

  string histoTag = "TimeBox";
  string histoName = histoTag
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str()
    + "_L" + layer.str()
    + "_wire" + wire.str();

  if (debug) cout<<"[DTAlbertoBenvenutiTask]: histoName "<<histoName<<endl;

  if ( parameters.getUntrackedParameter<bool>("readDB", false) ) 
    tTrigMap->slTtrig(dtWire.layerId().superlayerId(), tTrig, tTrigRMS); 
  else tTrig = parameters.getParameter<int>("defaultTtrig");
  
  string histoTitle = histoName + " (TDC Counts)";
  int timeBoxGranularity = parameters.getUntrackedParameter<int>("timeBoxGranularity",4);
  
  if (!parameters.getUntrackedParameter<bool>("readDB", true)) {
    int maxTDCCounts = 6400 * parameters.getUntrackedParameter<int>("tdcRescale", 1);
    TH1F *TB = new TH1F(histoName.c_str(),histoTitle.c_str(), maxTDCCounts/timeBoxGranularity, 0, maxTDCCounts);
    TBMap[dtWire] = TB;
  }    
  else {
    TH1F *TB = new TH1F(histoName.c_str(),histoTitle.c_str(), 2*tMax/timeBoxGranularity, tTrig-tMax, tTrig+2*tMax);
    TBMap[dtWire] = TB;
  }

}


void DTAlbertoBenvenutiTask::analyze(const edm::Event& e, const edm::EventSetup& c){
  
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

  int tdcCount = 0;
  DTDigiCollection::DigiRangeIterator dtLayerId_It;
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){
      tdcCount++;
    }
  }

  bool isSyncNoisy = false;
  if (tdcCount > maxTDCHits) isSyncNoisy = true;
  
  for (dtLayerId_It=dtdigis->begin(); dtLayerId_It!=dtdigis->end(); ++dtLayerId_It){
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerId_It).second).first;
	 digiIt!=((*dtLayerId_It).second).second; ++digiIt){
      
      bool isNoisy = false;
      bool isFEMasked = false;
      bool isTDCMasked = false;
      bool isTrigMask = false;
      bool isDead = false;
      bool isNohv = false;
      const DTWireId wireId(((*dtLayerId_It).first), (*digiIt).wire());
      if(checkNoisyChannels) {
	statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
      }      
 
      // for clearness..
      const  DTSuperLayerId dtSLId = ((*dtLayerId_It).first).superlayerId();
      uint32_t indexSL = dtSLId.rawId();
      const  DTChamberId dtChId = dtSLId.chamberId(); 
      uint32_t indexCh = dtChId.rawId();
      int layer_number=((*dtLayerId_It).first).layer();
      int superlayer_number=dtSLId.superlayer();
      const  DTLayerId dtLId = (*dtLayerId_It).first;
      uint32_t indexL = dtLId.rawId();
      
      float t0; float t0RMS;
      int tdcTime = (*digiIt).countsTDC();

      if (parameters.getParameter<bool>("performPerWireT0Calibration")) {
	const DTWireId dtWireId(((*dtLayerId_It).first), (*digiIt).wire());
	t0Map->cellT0(dtWireId, t0, t0RMS) ;
	tdcTime += int(round(t0));
      }
       
      // avoid to fill TB with noise
      if ((!isNoisy ) && (!isSyncNoisy)) {
	// TimeBoxes per wire
	if (TBMap.find(wireId) == TBMap.end()){
	  bookHistos(wireId);
	}
        TBMap[wireId]->Fill(tdcTime);
      }
    }
  }
}


