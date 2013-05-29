
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/03/10 16:54:12 $
 *  $Revision: 1.3 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/CalibMuon/interface/DTnoiseDBValidation.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// noise
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"


#include <stdio.h>
#include <sstream>
#include <math.h>
#include "TFile.h"
#include "TH1F.h"

using namespace edm;
using namespace std;




DTnoiseDBValidation::DTnoiseDBValidation(const ParameterSet& pset) {

  cout << "[DTnoiseDBValidation] Constructor called!" << endl;

  // Get the DQM needed services
  dbe = edm::Service<DQMStore>().operator->();
  dbe->setCurrentFolder("DT/noiseDBValidation");

  // Get dataBase label
  labelDBRef = pset.getUntrackedParameter<string>("labelDBRef");
  labelDB = pset.getUntrackedParameter<string>("labelDB");

  parameters = pset;
}


DTnoiseDBValidation::~DTnoiseDBValidation(){}

void DTnoiseDBValidation::beginRun(const edm::Run& run, const EventSetup& setup) {
  ESHandle<DTStatusFlag> noise_Ref;
  setup.get<DTStatusFlagRcd>().get(labelDBRef, noise_Ref);
  noiseRefMap = &*noise_Ref;
 
  ESHandle<DTStatusFlag> noise_toTest;
  setup.get<DTStatusFlagRcd>().get(labelDB, noise_toTest);
  noiseMap = &*noise_toTest;

  // Get the geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);
}

void DTnoiseDBValidation::beginJob() {


  metname = "noiseDbValidation";
  LogTrace(metname)<<"[DTnoiseDBValidation] Parameters initialization";
 
  outputFileName = parameters.getUntrackedParameter<std::string>("OutputFileName");

  noisyCells_Ref=0;
  noisyCells_toTest=0;

  // histo booking
  diffHisto = dbe->book1D("noisyCellDiff", "percentual (wrt the previous db) total number of noisy cells",1, 0.5, 1.5);
  diffHisto->setBinLabel(1,"diff");
  wheelHisto = dbe->book1D("wheelOccupancy", "percentual noisy cells occupancy per wheel",5, -2.5, 2.5);
  wheelHisto->setBinLabel(1,"wh-2");
  wheelHisto->setBinLabel(2,"wh-1");
  wheelHisto->setBinLabel(3,"wh0");
  wheelHisto->setBinLabel(4,"wh1");
  wheelHisto->setBinLabel(5,"wh2");
  stationHisto = dbe->book1D("stationOccupancy", "percentual noisy cells occupancy per station",4, 0.5, 4.5);
  stationHisto->setBinLabel(1,"st1");
  stationHisto->setBinLabel(2,"st2");
  stationHisto->setBinLabel(3,"st3");
  stationHisto->setBinLabel(4,"st4");
  sectorHisto = dbe->book1D("sectorOccupancy", "percentual noisy cells occupancy per sector",12, 0.5, 12.5);
  sectorHisto->setBinLabel(1,"sect1");
  sectorHisto->setBinLabel(2,"sect2");
  sectorHisto->setBinLabel(3,"sect3");
  sectorHisto->setBinLabel(4,"sect4");
  sectorHisto->setBinLabel(5,"sect5");
  sectorHisto->setBinLabel(6,"sect6");
  sectorHisto->setBinLabel(7,"sect7");
  sectorHisto->setBinLabel(8,"sect8");
  sectorHisto->setBinLabel(9,"sect9");
  sectorHisto->setBinLabel(10,"sect10");
  sectorHisto->setBinLabel(11,"sect11");
  sectorHisto->setBinLabel(12,"sect12");
  layerHisto = dbe->book1D("layerOccupancy", "percentual noisy cells occupancy per layer",3, 0.5, 3.5);
  layerHisto->setBinLabel(1,"first 10 bins");
  layerHisto->setBinLabel(2,"middle bins");
  layerHisto->setBinLabel(3,"last 10 bins");

  // map initialization
  map<int, int> whMap;
  whMap.clear();
  map<int, int> stMap;
  stMap.clear();
  map<int, int> sectMap;
  sectMap.clear();
  map<int, int> layerMap;
  layerMap.clear();

  // Loop over Ref DB entries
  for(DTStatusFlag::const_iterator noise = noiseRefMap->begin();
      noise != noiseRefMap->end(); noise++) {
    DTWireId wireId((*noise).first.wheelId,
		    (*noise).first.stationId,
		    (*noise).first.sectorId,
		    (*noise).first.slId,
		    (*noise).first.layerId,
		    (*noise).first.cellId);
    cout<< "Ref Wire: " <<  wireId<<endl;
    noisyCells_Ref++;
  }

  // Loop over Ref DB entries
  for(DTStatusFlag::const_iterator noise = noiseMap->begin();
      noise != noiseMap->end(); noise++) {
    DTWireId wireId((*noise).first.wheelId,
		    (*noise).first.stationId,
		    (*noise).first.sectorId,
		    (*noise).first.slId,
		    (*noise).first.layerId,
		    (*noise).first.cellId);
    cout<< "toTest Wire: " <<  wireId<<endl;
    noisyCells_toTest++;

    whMap[(*noise).first.wheelId]++;
    stMap[(*noise).first.stationId]++;
    sectMap[(*noise).first.sectorId]++;
    
    const DTTopology& dtTopo = dtGeom->layer(wireId.layerId())->specificTopology();
    const int lastWire = dtTopo.lastChannel();
    if((*noise).first.cellId<=10)
      layerMap[1]++;
    if((*noise).first.cellId>10 && (*noise).first.cellId<(lastWire-10))
      layerMap[2]++;
    if((*noise).first.cellId>=(lastWire-10))
      layerMap[3]++;
					 
  }

  //histo filling
  double scale = 1/double(noisyCells_Ref);
  diffHisto->Fill(1,abs(noisyCells_Ref-noisyCells_toTest)*scale);

  scale = 1/double(noisyCells_toTest);
  for(map<int, int >::const_iterator wheel = whMap.begin();
	wheel != whMap.end();
	wheel++) {
    wheelHisto->Fill((*wheel).first, ((*wheel).second)*scale);
  }
  for(map<int, int >::const_iterator station = stMap.begin();
	station != stMap.end();
	station++) {
    stationHisto->Fill((*station).first, ((*station).second)*scale);
  }
  for(map<int, int >::const_iterator sector = sectMap.begin();
	sector != sectMap.end();
	sector++) {
    sectorHisto->Fill((*sector).first, ((*sector).second)*scale);
  }
  for(map<int, int >::const_iterator layer = layerMap.begin();
	layer != layerMap.end();
	layer++) {
    layerHisto->Fill((*layer).first, ((*layer).second)*scale);
  }

}

void DTnoiseDBValidation::endJob() {

  // test on difference histo
  string testCriterionName = parameters.getUntrackedParameter<string>("diffTestName","noiseDifferenceInRange");
  const QReport * theDiffQReport = diffHisto->getQReport(testCriterionName);
  if(theDiffQReport) {
      vector<dqm::me_util::Channel> badChannels = theDiffQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	cout << " Bad partial difference of noisy channels! Contents : "<<(*channel).getContents()<<endl;
      }
  }
  testCriterionName = parameters.getUntrackedParameter<string>("wheelTestName","noiseWheelOccInRange");
  const QReport * theDiffQReport2 = wheelHisto->getQReport(testCriterionName);
  if(theDiffQReport2) {
      vector<dqm::me_util::Channel> badChannels = theDiffQReport2->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	int wheel = (*channel).getBin()-3;
	cout << " Bad percentual occupancy for wheel : "<<wheel<<"  Contents : "<<(*channel).getContents()<<endl;
      }
  }
  testCriterionName = parameters.getUntrackedParameter<string>("stationTestName","noiseStationOccInRange");
  const QReport * theDiffQReport3 = stationHisto->getQReport(testCriterionName);
  if(theDiffQReport3) {
      vector<dqm::me_util::Channel> badChannels = theDiffQReport3->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	cout << " Bad percentual occupancy for station : "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents()<<endl;
      }
  }
  testCriterionName = parameters.getUntrackedParameter<string>("sectorTestName","noiseSectorOccInRange");
  const QReport * theDiffQReport4 = sectorHisto->getQReport(testCriterionName);
  if(theDiffQReport4) {
      vector<dqm::me_util::Channel> badChannels = theDiffQReport4->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	cout << " Bad percentual occupancy for sector : "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents()<<endl;
      }
  }
  testCriterionName = parameters.getUntrackedParameter<string>("layerTestName","noiseLayerOccInRange");
  const QReport * theDiffQReport5 = layerHisto->getQReport(testCriterionName);
  if(theDiffQReport5) {
      vector<dqm::me_util::Channel> badChannels = theDiffQReport5->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	if((*channel).getBin()==1)
	  cout << " Bad percentual occupancy for the first 10 wires! Contents : "<<(*channel).getContents()<<endl;
	if((*channel).getBin()==2)
	  cout << " Bad percentual occupancy for the middle wires! Contents : "<<(*channel).getContents()<<endl;
	if((*channel).getBin()==3)
	  cout << " Bad percentual occupancy for the last 10 wires! Contents : "<<(*channel).getContents()<<endl;
      }
  }

  // write the histos on a file
  dbe->save(outputFileName);

}
