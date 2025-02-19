/*
 * \file DTtTrigCalibrationTest.cc
 * 
 * $Date: 2010/01/05 10:15:46 $
 * $Revision: 1.20 $
 * \author M. Zanetti - CERN
 * Modified by G. Mila - INFN Torino
 *
 */


#include "DQM/DTMonitorClient/src/DTtTrigCalibrationTest.h"

// Framework
#include <FWCore/Framework/interface/EventSetup.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// the Timebox fitter
#include "CalibMuon/DTCalibration/interface/DTTimeBoxFitter.h"

#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

DTtTrigCalibrationTest::DTtTrigCalibrationTest(const edm::ParameterSet& ps){
  
  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: Constructor";

  parameters = ps;
  
  dbe = edm::Service<DQMStore>().operator->();

  theFitter = new DTTimeBoxFitter();

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 3);

  percentual = parameters.getUntrackedParameter<int>("BadSLpercentual", 10);

}


DTtTrigCalibrationTest::~DTtTrigCalibrationTest(){

  edm::LogVerbatim ("tTrigCalibration") <<"DTtTrigCalibrationTest: analyzed " << nevents << " events";

  delete theFitter;

}


void DTtTrigCalibrationTest::beginJob(){

  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: BeginJob";

  nevents = 0;

}


void DTtTrigCalibrationTest::beginRun(Run const& run, EventSetup const& context) {

  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: BeginRun";

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

  for(map<int, MonitorElement*> ::const_iterator histo = wheelHistos.begin();
      histo != wheelHistos.end();
      histo++) {
    (*histo).second->Reset();
  }
  
  edm::LogVerbatim ("tTrigCalibration") <<"[DTtTrigCalibrationTest]: "<<nLumiSegs<<" updates";

  context.get<DTTtrigRcd>().get(tTrigMap);
  float tTrig, tTrigRMS,kFactor;

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

	TH1F * tb_histo_root = tb_histo->getTH1F();
	    
	pair<double, double> meanAndSigma = theFitter->fitTimeBox(tb_histo_root);
	    
        // ttrig and rms are counts
	tTrigMap->get(slID, tTrig, tTrigRMS, kFactor, DTTimeUnits::counts );

	if (histos.find((*ch_it)->id().rawId()) == histos.end()) bookHistos((*ch_it)->id());
	histos.find((*ch_it)->id().rawId())->second->setBinContent(slID.superLayer(), meanAndSigma.first-tTrig);

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
	  if(!((*ch_it)->id().station() == 4 && (*channel).getBin() == 3))
	    wheelHistos[(*ch_it)->id().wheel()]->Fill((*ch_it)->id().sector()-1,((*channel).getBin()-1)+3*((*ch_it)->id().station()-1));
	  else 
	    wheelHistos[(*ch_it)->id().wheel()]->Fill((*ch_it)->id().sector()-1,10);
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
	    wheelHistos[3]->Fill((*ch_it)->id().sector()-1,(*ch_it)->id().wheel());
	  }
	}
	// FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
	// edm::LogWarning ("tTrigCalibration") <<"-------- "<<theQReport->getMessage()<<" ------- "<<theQReport->getStatus();
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
    string histoName =  "t_TrigSummary_testFailedByAtLeastBadSL";
    wheelHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,0,14,5,-2,2);
    wheelHistos[3]->setBinLabel(1,"Sector1",1);
    wheelHistos[3]->setBinLabel(1,"Sector1",1);
    wheelHistos[3]->setBinLabel(2,"Sector2",1);
    wheelHistos[3]->setBinLabel(3,"Sector3",1);
    wheelHistos[3]->setBinLabel(4,"Sector4",1);
    wheelHistos[3]->setBinLabel(5,"Sector5",1);
    wheelHistos[3]->setBinLabel(6,"Sector6",1);
    wheelHistos[3]->setBinLabel(7,"Sector7",1);
    wheelHistos[3]->setBinLabel(8,"Sector8",1);
    wheelHistos[3]->setBinLabel(9,"Sector9",1);
    wheelHistos[3]->setBinLabel(10,"Sector10",1);
    wheelHistos[3]->setBinLabel(11,"Sector11",1);
    wheelHistos[3]->setBinLabel(12,"Sector12",1);
    wheelHistos[3]->setBinLabel(13,"Sector13",1);
    wheelHistos[3]->setBinLabel(14,"Sector14",1);
    wheelHistos[3]->setBinLabel(1,"Wheel-2",2);
    wheelHistos[3]->setBinLabel(2,"Wheel-1",2);
    wheelHistos[3]->setBinLabel(3,"Wheel0",2);
    wheelHistos[3]->setBinLabel(4,"Wheel+1",2);
    wheelHistos[3]->setBinLabel(5,"Wheel+2",2);
  }

  stringstream wheel; wheel <<wh;
  string histoName =  "t_TrigSummary_testFailed_W" + wheel.str();
  wheelHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,0,14,11,0,11);
  wheelHistos[wh]->setBinLabel(1,"Sector1",1);
  wheelHistos[wh]->setBinLabel(2,"Sector2",1);
  wheelHistos[wh]->setBinLabel(3,"Sector3",1);
  wheelHistos[wh]->setBinLabel(4,"Sector4",1);
  wheelHistos[wh]->setBinLabel(5,"Sector5",1);
  wheelHistos[wh]->setBinLabel(6,"Sector6",1);
  wheelHistos[wh]->setBinLabel(7,"Sector7",1);
  wheelHistos[wh]->setBinLabel(8,"Sector8",1);
  wheelHistos[wh]->setBinLabel(9,"Sector9",1);
  wheelHistos[wh]->setBinLabel(10,"Sector10",1);
  wheelHistos[wh]->setBinLabel(11,"Sector11",1);
  wheelHistos[wh]->setBinLabel(12,"Sector12",1);
  wheelHistos[wh]->setBinLabel(13,"Sector13",1);
  wheelHistos[wh]->setBinLabel(14,"Sector14",1);
  wheelHistos[wh]->setBinLabel(1,"MB1_SL1",2);
  wheelHistos[wh]->setBinLabel(2,"MB1_SL2",2);
  wheelHistos[wh]->setBinLabel(3,"MB1_SL3",2);
  wheelHistos[wh]->setBinLabel(4,"MB2_SL1",2);
  wheelHistos[wh]->setBinLabel(5,"MB2_SL2",2);
  wheelHistos[wh]->setBinLabel(6,"MB2_SL3",2);
  wheelHistos[wh]->setBinLabel(7,"MB3_SL1",2);
  wheelHistos[wh]->setBinLabel(8,"MB3_SL2",2);
  wheelHistos[wh]->setBinLabel(9,"MB3_SL3",2);
  wheelHistos[wh]->setBinLabel(10,"MB4_SL1",2);
  wheelHistos[wh]->setBinLabel(11,"MB4_SL3",2);

}
  
