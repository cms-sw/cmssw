

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/05/22 15:31:52 $
 *  $Revision: 1.7 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTEfficiencyTest.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DQMServices/Core/interface/MonitorElementBaseT.h>

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;

DTEfficiencyTest::DTEfficiencyTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("efficiency") << "[DTEfficiencyTest]: Constructor";

  parameters = ps;

  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);

}

DTEfficiencyTest::~DTEfficiencyTest(){

  edm::LogVerbatim ("efficiency") << "DTEfficiencyTest: analyzed " << nevents << " events";

}

void DTEfficiencyTest::endJob(){

  edm::LogVerbatim ("efficiency") << "[DTEfficiencyTest] endjob called!";

  dbe->rmdir("DT/Tests/DTEfficiency");

}

void DTEfficiencyTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("efficiency") << "[DTEfficiencyTest]: BeginJob";

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTEfficiencyTest::bookHistos(const DTLayerId & lId, int firstWire, int lastWire) {

  stringstream wheel; wheel << lId.superlayerId().wheel();
  stringstream station; station << lId.superlayerId().station();	
  stringstream sector; sector << lId.superlayerId().sector();
  stringstream superLayer; superLayer << lId.superlayerId().superlayer();
  stringstream layer; layer << lId.layer();

  string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() +  "_SL" + superLayer.str() +  "_L" + layer.str();
  string EfficiencyHistoName =  "Efficiency_" + HistoName; 
  string UnassEfficiencyHistoName =  "UnassEfficiency_" + HistoName; 

  dbe->setCurrentFolder("DT/Tests/DTEfficiency/Wheel" + wheel.str() +
			   "/Station" + station.str() +
			   "/Sector" + sector.str());

  EfficiencyHistos[HistoName] = dbe->book1D(EfficiencyHistoName.c_str(),EfficiencyHistoName.c_str(),lastWire-firstWire+1, firstWire-0.5, lastWire+0.5);
  UnassEfficiencyHistos[HistoName] = dbe->book1D(UnassEfficiencyHistoName.c_str(),UnassEfficiencyHistoName.c_str(),lastWire-firstWire+1, firstWire-0.5, lastWire+0.5);

}

void DTEfficiencyTest::analyze(const edm::Event& e, const edm::EventSetup& context){
  
  nevents++;

  edm::LogVerbatim ("efficiency") << "[DTEfficiencyTest]: "<<nevents<<" updates";

  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  edm::LogVerbatim ("efficiency") << "[DTEfficiencyTest]: Efficiency tests results";
  
  // Loop over the chambers
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId chID = (*ch_it)->id();
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();

    // Loop over the SuperLayers
    for(; sl_it != sl_end; ++sl_it) {
      DTSuperLayerId slID = (*sl_it)->id();
      vector<const DTLayer*>::const_iterator l_it = (*sl_it)->layers().begin();
      vector<const DTLayer*>::const_iterator l_end = (*sl_it)->layers().end();
      
      // Loop over the layers
      for(; l_it != l_end; ++l_it) {
	DTLayerId lID = (*l_it)->id();
	 
	stringstream wheel; wheel << chID.wheel();
	stringstream station; station << chID.station();
	stringstream sector; sector << chID.sector();
	stringstream superLayer; superLayer << slID.superlayer();
	stringstream layer; layer << lID.layer();
	
	string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() +  "_SL" + superLayer.str() +  "_L" + layer.str();
	
	// Get the ME produced by EfficiencyTask Source
	MonitorElement * occupancy_histo = dbe->get(getMEName("hEffOccupancy", lID));	
	MonitorElement * unassOccupancy_histo = dbe->get(getMEName("hEffUnassOccupancy", lID));
	MonitorElement * recSegmOccupancy_histo = dbe->get(getMEName("hRecSegmOccupancy", lID));
	 
	// ME -> TH1F
	if(occupancy_histo && unassOccupancy_histo && recSegmOccupancy_histo) {	  
	  MonitorElementT<TNamed>* occ = dynamic_cast<MonitorElementT<TNamed>*>(occupancy_histo);
	  MonitorElementT<TNamed>* unassOcc = dynamic_cast<MonitorElementT<TNamed>*>(unassOccupancy_histo);
	  MonitorElementT<TNamed>* recSegmOcc = dynamic_cast<MonitorElementT<TNamed>*>(recSegmOccupancy_histo);

	  if (occ && unassOcc && recSegmOcc) {
	    TH1F * occupancy_histo_root = dynamic_cast<TH1F*> (occ->operator->());
	    TH1F * unassOccupancy_histo_root = dynamic_cast<TH1F*> (unassOcc->operator->());
	    TH1F * recSegmOccupancy_histo_root = dynamic_cast<TH1F*> (recSegmOcc->operator->());

	    if (occupancy_histo_root && unassOccupancy_histo_root && recSegmOccupancy_histo_root) {	      
	      const int firstWire = muonGeom->layer(lID)->specificTopology().firstChannel();
	      const int lastWire = muonGeom->layer(lID)->specificTopology().lastChannel();

	      // Loop over the TH1F bin and fill the ME to be used for the Quality Test
	      for(int bin=firstWire; bin <= lastWire; bin++) {
		if((recSegmOccupancy_histo_root->GetBinContent(bin))!=0) {
		  //cout<<"book histos"<<endl;
		  if (EfficiencyHistos.find(HistoName) == EfficiencyHistos.end()) bookHistos(lID, firstWire, lastWire);
		  float efficiency = occupancy_histo_root->GetBinContent(bin) / recSegmOccupancy_histo_root->GetBinContent(bin);
		  float errorEff = sqrt(efficiency*(1-efficiency) / recSegmOccupancy_histo_root->GetBinContent(bin));
		  EfficiencyHistos.find(HistoName)->second->setBinContent(bin, efficiency);
		  EfficiencyHistos.find(HistoName)->second->setBinError(bin, errorEff);
		  
		  if (UnassEfficiencyHistos.find(HistoName) == EfficiencyHistos.end()) bookHistos(lID, firstWire, lastWire);
		  float unassEfficiency = unassOccupancy_histo_root->GetBinContent(bin) / recSegmOccupancy_histo_root->GetBinContent(bin);
		  float errorUnassEff = sqrt(unassEfficiency*(1-unassEfficiency) / recSegmOccupancy_histo_root->GetBinContent(bin));
		  UnassEfficiencyHistos.find(HistoName)->second->setBinContent(bin, unassEfficiency);	
		  UnassEfficiencyHistos.find(HistoName)->second->setBinError(bin, errorUnassEff);
		}
	      }
	      
	    }
	  }
	}
      } // loop on layers
    } // loop on superlayers
  } //loop on chambers
  
  // Efficiency test 
  //cout<<"[DTEfficiencyTest]: Efficiency Tests results"<<endl;
  string EfficiencyCriterionName = parameters.getUntrackedParameter<string>("EfficiencyTestName","EfficiencyInRange"); 
  for(map<string, MonitorElement*>::const_iterator hEff = EfficiencyHistos.begin();
      hEff != EfficiencyHistos.end();
      hEff++) {
    const QReport * theEfficiencyQReport = (*hEff).second->getQReport(EfficiencyCriterionName);
    if(theEfficiencyQReport) {
      vector<dqm::me_util::Channel> badChannels = theEfficiencyQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("efficiency") <<"LayerID : "<<(*hEff).first<< " Bad efficiency channels: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents();
      }
      edm::LogWarning ("efficiency") << "-------- "<<theEfficiencyQReport->getMessage()<<" ------- "<<theEfficiencyQReport->getStatus();
    }
  }
	
  // UnassEfficiency test 
  //cout<<"[DTEfficiencyTest]: UnassEfficiency Tests results"<<endl;
  string UnassEfficiencyCriterionName = parameters.getUntrackedParameter<string>("UnassEfficiencyTestName","UnassEfficiencyInRange"); 
  for(map<string, MonitorElement*>::const_iterator hUnassEff = UnassEfficiencyHistos.begin();
      hUnassEff != UnassEfficiencyHistos.end();
      hUnassEff++) {
    const QReport * theUnassEfficiencyQReport = (*hUnassEff).second->getQReport(UnassEfficiencyCriterionName);
    if(theUnassEfficiencyQReport) {
      vector<dqm::me_util::Channel> badChannels = theUnassEfficiencyQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("efficiency") << "Bad unassEfficiency channels: "<<(*channel).getBin()<<" "<<(*channel).getContents();
      }
      edm::LogWarning ("efficiency") << theUnassEfficiencyQReport->getMessage()<<" ------- "<<theUnassEfficiencyQReport->getStatus();
    }
  }

  
  if (nevents%parameters.getUntrackedParameter<int>("resultsSavingRate",10) == 0){
    if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) 
      dbe->save(parameters.getUntrackedParameter<string>("outputFile", "DTEfficiencyTest.root"));
    edm::LogStatistics();
  }
}


string DTEfficiencyTest::getMEName(string histoTag, const DTLayerId & lID) {

  stringstream wheel; wheel << lID.superlayerId().wheel();
  stringstream station; station << lID.superlayerId().station();
  stringstream sector; sector << lID.superlayerId().sector();
  stringstream superLayer; superLayer << lID.superlayerId().superlayer();
  stringstream layer; layer << lID.layer();

  string folderName = 
    "Collector/FU0/DT/DTEfficiencyTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + 
    "/SuperLayer" + superLayer.str() + "/";

  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str()
    + "_L" + layer.str();
  
  return histoname;
  
}
