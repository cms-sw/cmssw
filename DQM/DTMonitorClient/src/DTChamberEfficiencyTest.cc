/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
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

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

  nevents = 0;

  bookingdone = 0;

}



DTChamberEfficiencyTest::~DTChamberEfficiencyTest(){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "DTChamberEfficiencyTest: analyzed " << nevents << " events";

}

  void DTChamberEfficiencyTest::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, 
                                         edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {
  
  if (!bookingdone) {
  
  // Get the DT Geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // Loop over all the chambers
  vector<const DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();
  for (; ch_it != ch_end; ++ch_it) {
    // histo booking
    bookHistos(ibooker,(*ch_it)->id());
  }

  //summary histo booking
  bookHistos(ibooker);

  }

  bookingdone = 1; 
  
  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") <<"[DTChamberEfficiencyTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") <<"[DTChamberEfficiencyTest]: "<<nLumiSegs<<" updates";
  

  vector<const DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTChamberEfficiencyTest") << "[DTChamberEfficiencyTest]: ChamberEfficiency tests results"; 
  
  // Loop over the chambers
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId chID = (*ch_it)->id();
    
    stringstream wheel; wheel << chID.wheel();
    stringstream station; station << chID.station();
    stringstream sector; sector << chID.sector();
    
    string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
    
    // Get the ME produced by EfficiencyTask Source

    MonitorElement * GoodSegDen_histo = igetter.get(getMEName("hEffGoodSegVsPosDen", chID));	
    MonitorElement * GoodCloseSegNum_histo = igetter.get(getMEName("hEffGoodCloseSegVsPosNum", chID));
    
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

void DTChamberEfficiencyTest::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

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

void DTChamberEfficiencyTest::bookHistos(DQMStore::IBooker & ibooker, const DTChamberId & chId) {

  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();	
  stringstream sector; sector << chId.sector();

  string HistoName = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();
  string xEfficiencyHistoName =  "xEfficiency_" + HistoName; 
  string yEfficiencyHistoName =  "yEfficiency_" + HistoName; 
  string xVSyEffHistoName =  "xVSyEff_" + HistoName; 

  ibooker.setCurrentFolder("DT/01-DTChamberEfficiency/Wheel" + wheel.str() +
			"/Sector" + sector.str() +
                        "/Station" + station.str());

  xEfficiencyHistos[HistoName] = ibooker.book1D(xEfficiencyHistoName.c_str(),xEfficiencyHistoName.c_str(),25,-250.,250.);
  yEfficiencyHistos[HistoName] = ibooker.book1D(yEfficiencyHistoName.c_str(),yEfficiencyHistoName.c_str(),25,-250.,250.);
  xVSyEffHistos[HistoName] = ibooker.book2D(xVSyEffHistoName.c_str(),xVSyEffHistoName.c_str(),25,-250.,250., 25,-250.,250.);

}

void DTChamberEfficiencyTest::bookHistos(DQMStore::IBooker & ibooker) {

  for(int wh=-2; wh<=2; wh++){
    stringstream wheel; wheel << wh;
    string histoName =  "chEfficiencySummary_W" + wheel.str();

    ibooker.setCurrentFolder("DT/01-DTChamberEfficiency");
    summaryHistos[wh] = ibooker.book2D(histoName.c_str(),histoName.c_str(),12,1,13,4,1,5);
    summaryHistos[wh]->setAxisTitle("Sector",1);
    summaryHistos[wh]->setBinLabel(1,"MB1",2);
    summaryHistos[wh]->setBinLabel(2,"MB2",2);
    summaryHistos[wh]->setBinLabel(3,"MB3",2);
    summaryHistos[wh]->setBinLabel(4,"MB4",2);
  }

}
