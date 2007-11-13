/*
 * \file DTtTrigCalibrationTest.cc
 * 
 * $Date: 2007/11/07 15:29:35 $
 * $Revision: 1.10 $
 * \author M. Zanetti - CERN
 *
 */


#include "DQM/DTMonitorClient/src/DTtTrigCalibrationTest.h"

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

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// the Timebox fitter
#include "CalibMuon/DTCalibration/interface/DTTimeBoxFitter.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

DTtTrigCalibrationTest::DTtTrigCalibrationTest(const edm::ParameterSet& ps){
  
  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: Constructor";

  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);

  theFitter = new DTTimeBoxFitter();

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

  percentual = parameters.getUntrackedParameter<int>("BadSLpercentual", 10);

}


DTtTrigCalibrationTest::~DTtTrigCalibrationTest(){

  edm::LogVerbatim ("tTrigCalibration") <<"DTtTrigCalibrationTest: analyzed " << nevents << " events";

  delete theFitter;

}


void DTtTrigCalibrationTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: BeginJob";

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTtTrigCalibrationTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}


void DTtTrigCalibrationTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("tTrigCalibration") << "[DTtTrigCalibrationTest]: "<<nevents<<" events";

}



void DTtTrigCalibrationTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {


  // counts number of updats (online mode) or number of events (standalone mode)
  //nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  //if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
  //   nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;
  //edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: "<<nevents<<" updates";


  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: "<<nLumiSegs<<" updates";

  context.get<DTTtrigRcd>().get(tTrigMap);
  float tTrig, tTrigRMS;

  map <pair<int,int>, int> cmsHistos;
  cmsHistos.clear();
  map <pair<int,int>, bool> filled;
  for(int i=-2; i<3; i++){
    for(int j=1; j<15; j++){
      filled[make_pair(i,j)]=false;
    }
  }
  
  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    
    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();
    for(; sl_it != sl_end; ++sl_it) {
      
      DTSuperLayerId slID = (*sl_it)->id();
      
      MonitorElement * tb_histo = dbe->get(getMEName(slID));
      if (tb_histo) {
	
	edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: I've got the histo!!";	

	MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>(tb_histo);
	if (ob) {
	  TH1F * tb_histo_root = dynamic_cast<TH1F*> (ob->operator->());
	  if (tb_histo_root) {
	    
	    pair<double, double> meanAndSigma = theFitter->fitTimeBox(tb_histo_root);
	    
	    tTrigMap->slTtrig(slID, tTrig, tTrigRMS);

	    if (histos.find((*ch_it)->id().rawId()) == histos.end()) bookHistos((*ch_it)->id());
	    histos.find((*ch_it)->id().rawId())->second->setBinContent(slID.superLayer(), meanAndSigma.first-tTrig);

	  }
	}
      }
    }
    
    if (histos.find((*ch_it)->id().rawId()) != histos.end()) {
      string criterionName = parameters.getUntrackedParameter<string>("tTrigTestName","tTrigOffSet"); 
      const QReport * theQReport = histos.find((*ch_it)->id().rawId())->second->getQReport(criterionName);
      if(theQReport) {
	vector<dqm::me_util::Channel> badChannels = theQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	     channel != badChannels.end(); channel++) {
	  edm::LogError ("tTrigCalibration") <<"Chamber ID : "<<(*ch_it)->id()<<" Bad channels: "<<(*channel).getBin()<<" "<<(*channel).getContents();
	  if(wheelHistos.find((*ch_it)->id().wheel()) == wheelHistos.end()) bookHistos((*ch_it)->id(), (*ch_it)->id().wheel());
	  // fill the wheel summary histos if the SL has not passed the test
	  wheelHistos[(*ch_it)->id().wheel()]->Fill((*ch_it)->id().sector(),(*channel).getBin()+3*((*ch_it)->id().station()-1));
	  // fill the cms summary histo if the percentual of SL which have not passed the test 
	  // is more than a predefined treshold
	  cmsHistos[make_pair((*ch_it)->id().wheel(),(*ch_it)->id().sector())]++;
	  if(((*ch_it)->id().sector()<13 &&
	      double(cmsHistos[make_pair((*ch_it)->id().wheel(),(*ch_it)->id().sector())])/11>double(percentual)/100 &&
	      filled[make_pair((*ch_it)->id().wheel(),(*ch_it)->id().sector())]==false) ||
	     ((*ch_it)->id().sector()>=13 && 
	      double(cmsHistos[make_pair((*ch_it)->id().wheel(),(*ch_it)->id().sector())])/2>double(percentual)/100 &&
	      filled[make_pair((*ch_it)->id().wheel(),(*ch_it)->id().sector())]==false)){
	    filled[make_pair((*ch_it)->id().wheel(),(*ch_it)->id().sector())]=true;
	    wheelHistos[3]->Fill((*ch_it)->id().sector(),(*ch_it)->id().wheel());
	  }
	}
	edm::LogWarning ("tTrigCalibration") <<"-------- "<<theQReport->getMessage()<<" ------- "<<theQReport->getStatus();
      } 
    }

  }

}


void DTtTrigCalibrationTest::endJob(){

  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest] endjob called!";

  dbe->rmdir("DT/Tests/DTtTrigCalibration");
}




string DTtTrigCalibrationTest::getMEName(const DTSuperLayerId & slID) {

  stringstream wheel; wheel << slID.wheel();	
  stringstream station; station << slID.station();	
  stringstream sector; sector << slID.sector();	
  stringstream superLayer; superLayer << slID.superlayer();

  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderTag = parameters.getUntrackedParameter<string>("folderTag", "TimeBoxes");
  string folderName = 
    folderRoot + "DT/DTDigiTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/" + folderTag + "/";

  string histoTag = parameters.getUntrackedParameter<string>("histoTag", "TimeBox");
  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str(); 
  
  return histoname;
  
}



void DTtTrigCalibrationTest::bookHistos(const DTChamberId & ch) {

  stringstream wheel; wheel << ch.wheel();	
  stringstream station; station << ch.station();	
  stringstream sector; sector << ch.sector();	

  string histoName =  "tTrigTest_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str(); 

  dbe->setCurrentFolder("DT/Tests/DTtTrigCalibration");
  
  histos[ch.rawId()] = dbe->book1D(histoName.c_str(),histoName.c_str(),3,0,2);

}

void DTtTrigCalibrationTest::bookHistos(const DTChamberId & ch, int wh) {
  
  dbe->setCurrentFolder("DT/Tests/DTtTrigCalibration/SummaryPlot");

  if(wheelHistos.find(3) == wheelHistos.end()){
    string histoName =  "t_TrigSummary_testFailedByAtLeast%BadSL";
    wheelHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),15,0.5,14.5,6,-2.5,2.5);
  }

  stringstream wheel; wheel <<wh;
  string histoName =  "t_TrigSummary_testFailed_W" + wheel.str();
  wheelHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),15,0.5,14.5,13,0.5,12.5);

}
  
