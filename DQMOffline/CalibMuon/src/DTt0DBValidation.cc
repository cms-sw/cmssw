
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/04/18 12:29:24 $
 *  $Revision: 1.1 $
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
#include "CondFormats/DataRecord/interface/DTT0RefRcd.h"

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


void DTt0DBValidation::beginJob(const EventSetup& setup) {


  metname = "t0dbValidation";
  LogTrace(metname)<<"[DTt0DBValidation] Parameters initialization";
 
  outputFileName = parameters.getUntrackedParameter<std::string>("OutputFileName");

  ESHandle<DTT0> t0_Ref;
  setup.get<DTT0RefRcd>().get(labelDBRef, t0_Ref);
  tZeroRefMap = &*t0_Ref;
  LogTrace(metname)<<"[DTt0DBValidation] reference T0 version: " << t0_Ref->version();

  ESHandle<DTT0> t0;
  setup.get<DTT0Rcd>().get(labelDB, t0);
  tZeroMap = &*t0;
  LogTrace(metname)<<"[DTt0DBValidation] T0 to validate version: " << t0->version();

  // Get the geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Loop over Ref DB entries
  for(DTT0::const_iterator tzero = tZeroRefMap->begin();
      tzero != tZeroRefMap->end(); tzero++) {
    DTWireId wireId((*tzero).first.wheelId,
		    (*tzero).first.stationId,
		    (*tzero).first.sectorId,
		    (*tzero).first.slId,
		    (*tzero).first.layerId,
		    (*tzero).first.cellId);
    float t0mean = (*tzero).second.t0mean;
    float t0rms = (*tzero).second.t0rms;
    LogTrace(metname)<< "Ref Wire: " <<  wireId <<endl
		     << " T0 mean (TDC counts): " << t0mean
		     << " T0_rms (TDC counts): " << t0rms;

    t0RefMap[wireId].push_back(t0mean);
    t0RefMap[wireId].push_back(t0rms);
  }

  // Loop over Ref DB entries
  for(DTT0::const_iterator tzero = tZeroMap->begin();
      tzero != tZeroMap->end(); tzero++) {
    DTWireId wireId((*tzero).first.wheelId,
		    (*tzero).first.stationId,
		    (*tzero).first.sectorId,
		    (*tzero).first.slId,
		    (*tzero).first.layerId,
		    (*tzero).first.cellId);
    float t0mean = (*tzero).second.t0mean;
    float t0rms = (*tzero).second.t0rms;
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


