

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/01 00:39:51 $
 *  $Revision: 1.9 $
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

  edm::LogVerbatim ("chamberEfficiency") << "[DTChamberEfficiencyTest]: Constructor";

  parameters = ps;

  dbe = edm::Service<DQMStore>().operator->();
  dbe->setVerbose(1);

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

}



DTChamberEfficiencyTest::~DTChamberEfficiencyTest(){

  edm::LogVerbatim ("chamberEfficiency") << "DTChamberEfficiencyTest: analyzed " << nevents << " events";

}


void DTChamberEfficiencyTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("chamberEfficiency") <<"[DTChamberEfficiencyTest]: BeginJob"; 

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTChamberEfficiencyTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("chamberEfficiency") <<"[DTChamberEfficiencyTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}


void DTChamberEfficiencyTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("chamberEfficiency") << "[DTChamberEfficiencyTest]: "<<nevents<<" events";

}


void DTChamberEfficiencyTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  // counts number of updats (online mode) or number of events (standalone mode)
  //nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  //if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
  //     nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;  
  //edm::LogVerbatim ("chamberEfficiency") << "[DTChamberEfficiencyTest]: "<<nevents<<" updates";
  
  edm::LogVerbatim ("chamberEfficiency") <<"[DTChamberEfficiencyTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  edm::LogVerbatim ("chamberEfficiency") <<"[DTChamberEfficiencyTest]: "<<nLumiSegs<<" updates";
  

  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  edm::LogVerbatim ("chamberEfficiency") << "[DTChamberEfficiencyTest]: ChamberEfficiency tests results"; 
  
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
	  

      // Book Efficiency Histos
      if (xEfficiencyHistos.find(HistoName) == xEfficiencyHistos.end()) bookHistos(chID);
	  
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
	edm::LogError ("chamberEfficiency") << "Chamber : " << (*hXEff).first << " Bad XChamberEfficiency channels: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents();
      }
      // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
      // edm::LogWarning ("chamberEfficiency") << "-------- Chamber : "<<(*hXEff).first<<"  "<<theXEfficiencyQReport->getMessage()<<" ------- "<<theXEfficiencyQReport->getStatus();
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
	edm::LogError ("chamberEfficiency") << "Chamber : " << (*hYEff).first <<" Bad YChamberEfficiency channels: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents();
      }
      // FIXME: getMessage() sometimes returns and invalid string (null pointer inside QReport data member)
      // edm::LogWarning ("chamberEfficiency") << "-------- Chamber : "<<(*hYEff).first<<"  "<<theYEfficiencyQReport->getMessage()<<" ------- "<<theYEfficiencyQReport->getStatus();
    }
  }

}



void DTChamberEfficiencyTest::endJob(){

  edm::LogVerbatim ("chamberEfficiency") << "[DTChamberEfficiencyTest] endjob called!";
  dbe->rmdir("DT/Tests/DTChamberEfficiency");

}




string DTChamberEfficiencyTest::getMEName(string histoTag, const DTChamberId & chID) {

  stringstream wheel; wheel << chID.wheel();
  stringstream station; station << chID.station();
  stringstream sector; sector << chID.sector();
 
  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderName = 
    folderRoot + "DT/DTChamberEfficiencyTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/";

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

  dbe->setCurrentFolder("DT/Tests/DTChamberEfficiency/Wheel" + wheel.str() +
			   "/Station" + station.str() +
			   "/Sector" + sector.str());

  xEfficiencyHistos[HistoName] = dbe->book1D(xEfficiencyHistoName.c_str(),xEfficiencyHistoName.c_str(),25,-250.,250.);
  yEfficiencyHistos[HistoName] = dbe->book1D(yEfficiencyHistoName.c_str(),yEfficiencyHistoName.c_str(),25,-250.,250.);
  xVSyEffHistos[HistoName] = dbe->book2D(xVSyEffHistoName.c_str(),xVSyEffHistoName.c_str(),25,-250.,250., 25,-250.,250.);

}
