

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/05 10:15:45 $
 *  $Revision: 1.15 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTChamberEfficiencyTest.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Framework
#include <FWCore/Framework/interface/EventSetup.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdio.h>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;



DTChamberEfficiencyTest::DTChamberEfficiencyTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "[DTChamberEfficiencyTest]: Constructor";

  parameters = ps;

  dbe = edm::Service<DQMStore>().operator->();

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

}



DTChamberEfficiencyTest::~DTChamberEfficiencyTest(){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "DTChamberEfficiencyTest: analyzed " << nevents << " events";

}


void DTChamberEfficiencyTest::beginJob(){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") <<"[DTChamberEfficiencyTest]: BeginJob"; 

  nevents = 0;

}


void DTChamberEfficiencyTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") <<"[DTChamberEfficiencyTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}


void DTChamberEfficiencyTest::beginRun(const edm::Run& run, const edm::EventSetup& setup){
  
  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(muonGeom);

  // Loop over all the chambers
  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    // histo booking
    bookHistos((*ch_it)->id());
  }

  //summary histo booking
  bookHistos();
}

void DTChamberEfficiencyTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "[DTChamberEfficiencyTest]: "<<nevents<<" events";

}


void DTChamberEfficiencyTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  // counts number of updats (online mode) or number of events (standalone mode)
  //nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  //if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
  //     nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;  
  //edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "[DTChamberEfficiencyTest]: "<<nevents<<" updates";
  
  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") <<"[DTChamberEfficiencyTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") <<"[DTChamberEfficiencyTest]: "<<nLumiSegs<<" updates";
  

  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "[DTChamberEfficiencyTest]: ChamberEfficiency tests results"; 
  
  // Loop over the chambers
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId chID = (*ch_it)->id();
    
    stringstream wheel; wheel << chID.wheel();
    stringstream station; station << chID.station();
    stringstream sector; sector << chID.sector();
    
    string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
    
    // Get the ME produced by EfficiencyTask Source
    MonitorElement * GoodSegDen_histo = dbe->get(getMEName("hEffGoodSegVsPosDen", chID));	
    MonitorElement * GoodCloseSegNum_histo = dbe->get(getMEName("hEffGoodCloseSegVsPosNum", chID));
    
    // ME -> TH1F
    if(GoodSegDen_histo && GoodCloseSegNum_histo) {	  
      TH2F * GoodSegDen_histo_root = GoodSegDen_histo->getTH2F();
      TH2F * GoodCloseSegNum_histo_root = GoodCloseSegNum_histo->getTH2F();
	
      int lastBinX=(*GoodSegDen_histo_root).GetNbinsX();
      TH1D* proxN=GoodCloseSegNum_histo_root->ProjectionX();
      TH1D* proxD=GoodSegDen_histo_root->ProjectionX();

      int lastBinY=(*GoodSegDen_histo_root).GetNbinsY();
      TH1D* proyN=GoodCloseSegNum_histo_root->ProjectionY();
      TH1D* proyD=GoodSegDen_histo_root->ProjectionY();
	  
      for(int xBin=1; xBin<=lastBinX; xBin++) {
	if(proxD->GetBinContent(xBin)!=0){
	  float Xefficiency = proxN->GetBinContent(xBin) / proxD->GetBinContent(xBin);
	  xEfficiencyHistos.find(HistoName)->second->setBinContent(xBin, Xefficiency);
	}

	for(int yBin=1; yBin<=lastBinY; yBin++) {
	  if(GoodSegDen_histo_root->GetBinContent(xBin, yBin)!=0){
	    float XvsYefficiency = GoodCloseSegNum_histo_root->GetBinContent(xBin, yBin) / GoodSegDen_histo_root->GetBinContent(xBin, yBin);
	    xVSyEffHistos.find(HistoName)->second->setBinContent(xBin, yBin, XvsYefficiency);
	  }
	}
	    
      }
	  
      for(int yBin=1; yBin<=lastBinY; yBin++) {
	if(proyD->GetBinContent(yBin)!=0){
	  float Yefficiency = proyN->GetBinContent(yBin) / proyD->GetBinContent(yBin);
	  yEfficiencyHistos.find(HistoName)->second->setBinContent(yBin, Yefficiency);
	}
      }
    }
  } // loop on chambers
  
  
  // ChamberEfficiency test on X axis
  string XEfficiencyCriterionName = parameters.getUntrackedParameter<string>("XEfficiencyTestName","ChEfficiencyInRangeX"); 
  for(map<string, MonitorElement*>::const_iterator hXEff = xEfficiencyHistos.begin();
      hXEff != xEfficiencyHistos.end();
      hXEff++) {
    const QReport * theXEfficiencyQReport = (*hXEff).second->getQReport(XEfficiencyCriterionName);
    if(theXEfficiencyQReport) {
      vector<dqm::me_util::Channel> badChannels = theXEfficiencyQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "Chamber : " << (*hXEff).first << " Bad XChamberEfficiency channels: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents();
      }
      // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
      // edm::LogWarning ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "-------- Chamber : "<<(*hXEff).first<<"  "<<theXEfficiencyQReport->getMessage()<<" ------- "<<theXEfficiencyQReport->getStatus();
    }
  }


  // ChamberEfficiency test on Y axis
  string YEfficiencyCriterionName = parameters.getUntrackedParameter<string>("YEfficiencyTestName","ChEfficiencyInRangeY"); 
  for(map<string, MonitorElement*>::const_iterator hYEff = yEfficiencyHistos.begin();
      hYEff != yEfficiencyHistos.end();
      hYEff++) {
    const QReport * theYEfficiencyQReport = (*hYEff).second->getQReport(YEfficiencyCriterionName);
    if(theYEfficiencyQReport) {
      vector<dqm::me_util::Channel> badChannels = theYEfficiencyQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "Chamber : " << (*hYEff).first <<" Bad YChamberEfficiency channels: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents();
      }
      // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
      // edm::LogWarning ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "-------- Chamber : "<<(*hYEff).first<<"  "<<theYEfficiencyQReport->getMessage()<<" ------- "<<theYEfficiencyQReport->getStatus();
    }
  }

  //Fill the report summary histos
  for(int wh=-2; wh<=2; wh++){
    for(int sec=1; sec<=12; sec++){
      for(int st=1; st<=4; st++){

	summaryHistos[wh]->Fill(sec,st,1);
      
      }
    }
  }

}



void DTChamberEfficiencyTest::endJob(){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "[DTChamberEfficiencyTest] endjob called!";

}




string DTChamberEfficiencyTest::getMEName(string histoTag, const DTChamberId & chID) {

  stringstream wheel; wheel << chID.wheel();
  stringstream station; station << chID.station();
  stringstream sector; sector << chID.sector();
 
  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderName = 
    folderRoot + "DT/01-DTChamberEfficiency/Task/Wheel" +  wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/";

  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str();
  
  return histoname;
  
}


void DTChamberEfficiencyTest::bookHistos(const DTChamberId & chId) {

  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();	
  stringstream sector; sector << chId.sector();

  string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
  string xEfficiencyHistoName =  "xEfficiency_" + HistoName; 
  string yEfficiencyHistoName =  "yEfficiency_" + HistoName; 
  string xVSyEffHistoName =  "xVSyEff_" + HistoName; 

  dbe->setCurrentFolder("DT/01-DTChamberEfficiency/Wheel" + wheel.str() +
			"/Sector" + sector.str() +
                        "/Station" + station.str());

  xEfficiencyHistos[HistoName] = dbe->book1D(xEfficiencyHistoName.c_str(),xEfficiencyHistoName.c_str(),25,-250.,250.);
  yEfficiencyHistos[HistoName] = dbe->book1D(yEfficiencyHistoName.c_str(),yEfficiencyHistoName.c_str(),25,-250.,250.);
  xVSyEffHistos[HistoName] = dbe->book2D(xVSyEffHistoName.c_str(),xVSyEffHistoName.c_str(),25,-250.,250., 25,-250.,250.);

}


void DTChamberEfficiencyTest::bookHistos() {

  for(int wh=-2; wh<=2; wh++){
    stringstream wheel; wheel << wh;
    string histoName =  "chEfficiencySummary_W" + wheel.str();
    dbe->setCurrentFolder("DT/01-DTChamberEfficiency");
    summaryHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),12,1,13,4,1,5);
    summaryHistos[wh]->setAxisTitle("Sector",1);
    summaryHistos[wh]->setBinLabel(1,"MB1",2);
    summaryHistos[wh]->setBinLabel(2,"MB2",2);
    summaryHistos[wh]->setBinLabel(3,"MB3",2);
    summaryHistos[wh]->setBinLabel(4,"MB4",2);
  }

}
