
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/02/16 15:57:24 $
 *  $Revision: 1.6 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/CalibMuon/interface/DTt0DBValidation.h"

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

// t0
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

#include <stdio.h>
#include <sstream>
#include <math.h>
#include "TFile.h"

using namespace edm;
using namespace std;




DTt0DBValidation::DTt0DBValidation(const ParameterSet& pset) {

  cout << "[DTt0DBValidation] Constructor called!" << endl;

  // Get the DQM needed services
  dbe = edm::Service<DQMStore>().operator->();
  dbe->setCurrentFolder("DT/DTDBValidation");

  // Get dataBase label
  labelDBRef = pset.getUntrackedParameter<string>("labelDBRef");
  labelDB = pset.getUntrackedParameter<string>("labelDB");

  parameters = pset;
}


DTt0DBValidation::~DTt0DBValidation(){}


void DTt0DBValidation::beginRun(const edm::Run& run, const EventSetup& setup) {

  metname = "t0dbValidation";
  LogTrace(metname)<<"[DTt0DBValidation] Parameters initialization";
 
  outputFileName = parameters.getUntrackedParameter<std::string>("OutputFileName");

  ESHandle<DTT0> t0_Ref;
  setup.get<DTT0Rcd>().get(labelDBRef, t0_Ref);
  tZeroRefMap = &*t0_Ref;
  LogTrace(metname)<<"[DTt0DBValidation] reference T0 version: " << t0_Ref->version();

  ESHandle<DTT0> t0;
  setup.get<DTT0Rcd>().get(labelDB, t0);
  tZeroMap = &*t0;
  LogTrace(metname)<<"[DTt0DBValidation] T0 to validate version: " << t0->version();

  //book&reset the summary histos
  for(int wheel=-2; wheel<=2; wheel++){
    bookHistos(wheel);
    wheelSummary[wheel]->Reset();
  }

  // Get the geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Loop over Ref DB entries
  for(DTT0::const_iterator tzero = tZeroRefMap->begin();
      tzero != tZeroRefMap->end(); tzero++) {
    // t0s and rms are TDC counts
    DTWireId wireId((*tzero).first.wheelId,
		    (*tzero).first.stationId,
		    (*tzero).first.sectorId,
		    (*tzero).first.slId,
		    (*tzero).first.layerId,
		    (*tzero).first.cellId);
    float t0mean;
    float t0rms;
    tZeroRefMap->get( wireId, t0mean, t0rms, DTTimeUnits::counts );
    LogTrace(metname)<< "Ref Wire: " <<  wireId <<endl
		     << " T0 mean (TDC counts): " << t0mean
		     << " T0_rms (TDC counts): " << t0rms;

    t0RefMap[wireId].push_back(t0mean);
    t0RefMap[wireId].push_back(t0rms);
  }

  // Loop over Ref DB entries
  for(DTT0::const_iterator tzero = tZeroMap->begin();
      tzero != tZeroMap->end(); tzero++) {
    // t0s and rms are TDC counts
    DTWireId wireId((*tzero).first.wheelId,
		    (*tzero).first.stationId,
		    (*tzero).first.sectorId,
		    (*tzero).first.slId,
		    (*tzero).first.layerId,
		    (*tzero).first.cellId);
    float t0mean;
    float t0rms;
    tZeroMap->get( wireId, t0mean, t0rms, DTTimeUnits::counts );
    LogTrace(metname)<< "Wire: " <<  wireId <<endl
		     << " T0 mean (TDC counts): " << t0mean
		     << " T0_rms (TDC counts): " << t0rms;

    t0Map[wireId].push_back(t0mean);
    t0Map[wireId].push_back(t0rms);
  }

  double difference=0;
  for(map<DTWireId, vector<float> >::const_iterator theMap = t0RefMap.begin();
      theMap != t0RefMap.end();
      theMap++) {  
    if(t0Map.find((*theMap).first) != t0Map.end()) {

      // compute the difference
      difference = t0Map[(*theMap).first][0]-(*theMap).second[0];

      //book histo
      DTLayerId layerId = (*theMap).first.layerId();
      if(t0DiffHistos.find(layerId) == t0DiffHistos.end()) {
	const DTTopology& dtTopo = dtGeom->layer(layerId)->specificTopology();
	const int firstWire = dtTopo.firstChannel();
	const int lastWire = dtTopo.lastChannel();
	bookHistos(layerId, firstWire, lastWire);
      }

      cout<< "Filling the histo for wire: "<<(*theMap).first
	  <<"  difference: "<<difference<<endl;
      t0DiffHistos[layerId]->Fill((*theMap).first.wire(),difference);

    }
  } // Loop over the t0 map reference
   
}


void DTt0DBValidation::endJob() {  

  //check the histos
  string testCriterionName = parameters.getUntrackedParameter<string>("t0TestName","t0DifferenceInRange"); 
  for(map<DTLayerId, MonitorElement*>::const_iterator hDiff = t0DiffHistos.begin();
      hDiff != t0DiffHistos.end();
      hDiff++) {
    const QReport * theDiffQReport = (*hDiff).second->getQReport(testCriterionName);
    if(theDiffQReport) {
      vector<dqm::me_util::Channel> badChannels = theDiffQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	cout << "layer:"<<(*hDiff).first<<" Bad mean channels: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents()<<endl;

	int xBin = ((*hDiff).first.station()-1)*12+(*hDiff).first.layer()+4*((*hDiff).first.superlayer()-1);
	if((*hDiff).first.station()==4 && (*hDiff).first.superlayer()==3)
	  xBin = ((*hDiff).first.station()-1)*12+(*hDiff).first.layer()+4*((*hDiff).first.superlayer()-2);
	wheelSummary[(*hDiff).first.wheel()]->Fill(xBin,(*hDiff).first.sector());
 
      }
      cout << "-------- layer: "<<(*hDiff).first<<"  "<<theDiffQReport->getMessage()<<" ------- "<<theDiffQReport->getStatus()<<endl; 
    }
  }

  // write the histos on a file
  dbe->save(outputFileName);

}

  // Book a set of histograms for a given Layer
void DTt0DBValidation::bookHistos(DTLayerId lId, int firstWire, int lastWire) {
  
  LogTrace(metname)<< "   Booking histos for L: " << lId;

  // Compose the chamber name
  stringstream wheel; wheel << lId.superlayerId().chamberId().wheel();	
  stringstream station; station << lId.superlayerId().chamberId().station();	
  stringstream sector; sector << lId.superlayerId().chamberId().sector();	
  stringstream superLayer; superLayer << lId.superlayerId().superlayer();	
  stringstream layer; layer << lId.layer();

  string lHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str() +
    "_SL" + superLayer.str()+
    "_L" + layer.str();
  
  dbe->setCurrentFolder("DT/t0Validation/Wheel" + wheel.str() +
			   "/Station" + station.str() +
			   "/Sector" + sector.str() +
			   "/SuperLayer" +superLayer.str());
  // Create the monitor elements
  MonitorElement * hDifference;
  hDifference = dbe->book1D("hDifference"+lHistoName, "difference between the two t0 values",lastWire-firstWire+1, firstWire-0.5, lastWire+0.5);
  
  t0DiffHistos[lId] = hDifference;
}

// Book the summary histos
void DTt0DBValidation::bookHistos(int wheel) {
  dbe->setCurrentFolder("DT/t0Validation/Summary");
  stringstream wh; wh << wheel;
    wheelSummary[wheel]= dbe->book2D("summaryWrongT0_W"+wh.str(), "W"+wh.str()+": summary of wrong t0 differences",44,1,45,14,1,15);
    wheelSummary[wheel]->setBinLabel(1,"M1L1",1);
    wheelSummary[wheel]->setBinLabel(2,"M1L2",1);
    wheelSummary[wheel]->setBinLabel(3,"M1L3",1);
    wheelSummary[wheel]->setBinLabel(4,"M1L4",1);
    wheelSummary[wheel]->setBinLabel(5,"M1L5",1);
    wheelSummary[wheel]->setBinLabel(6,"M1L6",1);
    wheelSummary[wheel]->setBinLabel(7,"M1L7",1);
    wheelSummary[wheel]->setBinLabel(8,"M1L8",1);
    wheelSummary[wheel]->setBinLabel(9,"M1L9",1);
    wheelSummary[wheel]->setBinLabel(10,"M1L10",1);
    wheelSummary[wheel]->setBinLabel(11,"M1L11",1);
    wheelSummary[wheel]->setBinLabel(12,"M1L12",1);
    wheelSummary[wheel]->setBinLabel(13,"M2L1",1);
    wheelSummary[wheel]->setBinLabel(14,"M2L2",1);
    wheelSummary[wheel]->setBinLabel(15,"M2L3",1);
    wheelSummary[wheel]->setBinLabel(16,"M2L4",1);
    wheelSummary[wheel]->setBinLabel(17,"M2L5",1);
    wheelSummary[wheel]->setBinLabel(18,"M2L6",1);
    wheelSummary[wheel]->setBinLabel(19,"M2L7",1);
    wheelSummary[wheel]->setBinLabel(20,"M2L8",1);
    wheelSummary[wheel]->setBinLabel(21,"M2L9",1);
    wheelSummary[wheel]->setBinLabel(22,"M2L10",1);
    wheelSummary[wheel]->setBinLabel(23,"M2L11",1);
    wheelSummary[wheel]->setBinLabel(24,"M2L12",1);
    wheelSummary[wheel]->setBinLabel(25,"M3L1",1);
    wheelSummary[wheel]->setBinLabel(26,"M3L2",1);
    wheelSummary[wheel]->setBinLabel(27,"M3L3",1);
    wheelSummary[wheel]->setBinLabel(28,"M3L4",1);
    wheelSummary[wheel]->setBinLabel(29,"M3L5",1);
    wheelSummary[wheel]->setBinLabel(30,"M3L6",1);
    wheelSummary[wheel]->setBinLabel(31,"M3L7",1);
    wheelSummary[wheel]->setBinLabel(32,"M3L8",1);
    wheelSummary[wheel]->setBinLabel(33,"M3L9",1);
    wheelSummary[wheel]->setBinLabel(34,"M3L10",1);
    wheelSummary[wheel]->setBinLabel(35,"M3L11",1);
    wheelSummary[wheel]->setBinLabel(36,"M3L12",1);
    wheelSummary[wheel]->setBinLabel(37,"M4L1",1);
    wheelSummary[wheel]->setBinLabel(38,"M4L2",1);
    wheelSummary[wheel]->setBinLabel(39,"M4L3",1);
    wheelSummary[wheel]->setBinLabel(40,"M4L4",1);
    wheelSummary[wheel]->setBinLabel(41,"M4L5",1);
    wheelSummary[wheel]->setBinLabel(42,"M4L6",1);
    wheelSummary[wheel]->setBinLabel(43,"M4L7",1);
    wheelSummary[wheel]->setBinLabel(44,"M4L8",1);
}
