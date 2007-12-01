

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/11/07 15:29:24 $
 *  $Revision: 1.14 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTResolutionTest.h>

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


DTResolutionTest::DTResolutionTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("resolution") << "[DTResolutionTest]: Constructor";
  parameters = ps;

  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

  percentual = parameters.getUntrackedParameter<int>("BadSLpercentual", 10);

}


DTResolutionTest::~DTResolutionTest(){

  edm::LogVerbatim ("resolution") << "DTResolutionTest: analyzed " << nevents << " events";

}


void DTResolutionTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("resolution") <<"[DTResolutionTest]: BeginJob"; 

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTResolutionTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("resolution") <<"[DTResolutionTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}


void DTResolutionTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("resolution") << "[DTResolutionTest]: "<<nevents<<" events";

}



void DTResolutionTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  // counts number of updats (online mode) or number of events (standalone mode)
  //nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  //if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
  //   nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;
  //edm::LogVerbatim ("resolution") << "[DTResolutionTest]: "<<nevents<<" updates";

  edm::LogVerbatim ("resolution") <<"[DTResolutionTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  for(map<int, MonitorElement*> ::const_iterator histo = wheelMeanHistos.begin();
      histo != wheelMeanHistos.end();
      histo++) {
    (*histo).second->Reset();
  }
  for(map<int, MonitorElement*> ::const_iterator histo = wheelSigmaHistos.begin();
      histo != wheelSigmaHistos.end();
      histo++) {
    (*histo).second->Reset();
  }
  map <pair<int,int>, int> cmsMeanHistos;
  cmsMeanHistos.clear();
  map <pair<int,int>, bool> MeanFilled;
  for(int i=-2; i<3; i++){
    for(int j=1; j<15; j++){
      MeanFilled[make_pair(i,j)]=false;
    }
  }
  map <pair<int,int>, int> cmsSigmaHistos;
  cmsSigmaHistos.clear();
  map <pair<int,int>, bool> SigmaFilled;
  for(int i=-2; i<3; i++){
    for(int j=1; j<15; j++){
      SigmaFilled[make_pair(i,j)]=false;
    }
  }

  edm::LogVerbatim ("resolution") <<"[DTResolutionTest]: "<<nLumiSegs<<" updates";

  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  edm::LogVerbatim ("resolution") << "[DTResolutionTest]: Residual Distribution tests results";
  
  for (; ch_it != ch_end; ++ch_it) {

    DTChamberId chID = (*ch_it)->id();

    // Fill the test histos
    int entry=-1;
    if(chID.station() == 1) entry=0;
    if(chID.station() == 2) entry=3;
    if(chID.station() == 3) entry=6;
    if(chID.station() == 4) entry=9;

    vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin(); 
    vector<const DTSuperLayer*>::const_iterator sl_end = (*ch_it)->superLayers().end();

    for(; sl_it != sl_end; ++sl_it) {

      DTSuperLayerId slID = (*sl_it)->id();

      stringstream wheel; wheel << slID.wheel();	
      stringstream station; station << slID.station();	
      stringstream sector; sector << slID.sector();	
      stringstream superLayer; superLayer << slID.superlayer();
      
      string HistoName = "W" + wheel.str() + "_Sec" + sector.str(); 
      string supLayer = "W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str() + "_SL" + superLayer.str(); 

      MonitorElement * res_histo = dbe->get(getMEName(slID));
      if (res_histo) {
	// gaussian test
	string GaussianCriterionName = 
	  parameters.getUntrackedParameter<string>("resDistributionTestName",
						   "ResidualsDistributionGaussianTest");
	const QReport * GaussianReport = res_histo->getQReport(GaussianCriterionName);
	if(GaussianReport){
	  edm::LogWarning ("resolution") << "-------- SuperLayer : "<<supLayer<<"  "<<GaussianReport->getMessage()<<" ------- "<<GaussianReport->getStatus();
	}
	int BinNumber = entry+slID.superLayer();
	if(BinNumber == 12) BinNumber=11;
	float mean = (*res_histo).getMean(1);
	float sigma = (*res_histo).getRMS(1);
	if (MeanHistos.find(make_pair(slID.wheel(),slID.sector())) == MeanHistos.end()) bookHistos((*ch_it)->id());
	MeanHistos.find(make_pair(slID.wheel(),slID.sector()))->second->setBinContent(BinNumber, mean);	
	SigmaHistos.find(make_pair(slID.wheel(),slID.sector()))->second->setBinContent(BinNumber, sigma);
      }
    }
  }

  // Mean test 
  string MeanCriterionName = parameters.getUntrackedParameter<string>("meanTestName","ResidualsMeanInRange"); 
  for(map<pair<int,int>, MonitorElement*>::const_iterator hMean = MeanHistos.begin();
      hMean != MeanHistos.end();
      hMean++) {
    const QReport * theMeanQReport = (*hMean).second->getQReport(MeanCriterionName);
    stringstream wheel; wheel << (*hMean).first.first;
    stringstream sector; sector << (*hMean).first.second;
    if(theMeanQReport) {
      vector<dqm::me_util::Channel> badChannels = theMeanQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("resolution") << "wheel: "<<wheel.str()<<" sector: "<<sector.str()<<" Bad mean channels: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents();
	string HistoName = "W" + wheel.str() + "_Sec" + sector.str();
	if (MeanHistosSetRange.find(HistoName) == MeanHistosSetRange.end()) bookHistos((*ch_it)->id());
	MeanHistosSetRange.find(HistoName)->second->Fill((*channel).getBin());
	if (MeanHistosSetRange2D.find(HistoName) == MeanHistosSetRange2D.end()) bookHistos((*ch_it)->id());
        MeanHistosSetRange2D.find(HistoName)->second->Fill((*channel).getBin(),(*channel).getContents());
	if(wheelMeanHistos.find((*hMean).first.first) == wheelMeanHistos.end()) bookHistos((*hMean).first.first);
	// fill the wheel summary histos if the SL has not passed the test
	wheelMeanHistos[(*hMean).first.first]->Fill(((*hMean).first.second)-1,(*channel).getBin()-1);
	// fill the cms summary histo if the percentual of SL which have not passed the test 
	// is more than a predefined treshold
	cmsMeanHistos[make_pair((*hMean).first.first,(*hMean).first.second)]++;
	if(((*hMean).first.second<13 &&
	    double(cmsMeanHistos[make_pair((*hMean).first.first,(*hMean).first.second)])/11>double(percentual)/100 &&
	    MeanFilled[make_pair((*hMean).first.first,(*hMean).first.second)]==false) ||
	   ((*hMean).first.first>=13 && 
	    double(cmsMeanHistos[make_pair((*hMean).first.first,(*hMean).first.second)])/2>double(percentual)/100 &&
	    MeanFilled[make_pair((*hMean).first.first,(*hMean).first.second)]==false)){
	  MeanFilled[make_pair((*hMean).first.first,(*hMean).first.second)]=true;
	  wheelMeanHistos[3]->Fill(((*hMean).first.second)-1,(*hMean).first.first);
	}	
      }
      edm::LogWarning ("resolution") << "-------- wheel: "<<wheel.str()<<" sector: "<<sector.str()<<"  "<<theMeanQReport->getMessage()<<" ------- "<<theMeanQReport->getStatus(); 
    }
  }

  // Sigma test
  string SigmaCriterionName = parameters.getUntrackedParameter<string>("sigmaTestName","ResidualsSigmaInRange"); 
  for(map<pair<int,int>, MonitorElement*>::const_iterator hSigma = SigmaHistos.begin();
      hSigma != SigmaHistos.end();
      hSigma++) {
    const QReport * theSigmaQReport = (*hSigma).second->getQReport(SigmaCriterionName);
    stringstream wheel; wheel << (*hSigma).first.first;
    stringstream sector; sector << (*hSigma).first.second;
    if(theSigmaQReport) {
      vector<dqm::me_util::Channel> badChannels = theSigmaQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("resolution") << "wheel: "<<wheel.str()<<" sector: "<<sector.str()<<" Bad sigma channels: "<<(*channel).getBin()<<" Contents : "<<(*channel).getContents();
	string HistoName = "W" + wheel.str() + "_Sec" + sector.str();
	if (SigmaHistosSetRange.find(HistoName) == SigmaHistosSetRange.end()) bookHistos((*ch_it)->id());
        SigmaHistosSetRange.find(HistoName)->second->Fill((*channel).getBin());
        if (SigmaHistosSetRange2D.find(HistoName) == SigmaHistosSetRange2D.end()) bookHistos((*ch_it)->id());
        SigmaHistosSetRange2D.find(HistoName)->second->Fill((*channel).getBin(),(*channel).getContents());
	if(wheelSigmaHistos.find((*hSigma).first.first) == wheelSigmaHistos.end()) bookHistos((*hSigma).first.first);
	// fill the wheel summary histos if the SL has not passed the test
	wheelSigmaHistos[(*hSigma).first.first]->Fill(((*hSigma).first.second)-1,(*channel).getBin()-1);
	// fill the cms summary histo if the percentual of SL which have not passed the test 
	// is more than a predefined treshold
	cmsSigmaHistos[make_pair((*hSigma).first.first,(*hSigma).first.second)]++;
	if(((*hSigma).first.second<13 &&
	    double(cmsSigmaHistos[make_pair((*hSigma).first.first,(*hSigma).first.second)])/11>double(percentual)/100 &&
	    SigmaFilled[make_pair((*hSigma).first.first,(*hSigma).first.second)]==false) ||
	   ((*hSigma).first.first>=13 && 
	    double(cmsSigmaHistos[make_pair((*hSigma).first.first,(*hSigma).first.second)])/2>double(percentual)/100 &&
	    SigmaFilled[make_pair((*hSigma).first.first,(*hSigma).first.second)]==false)){
	  SigmaFilled[make_pair((*hSigma).first.first,(*hSigma).first.second)]=true;
	  wheelSigmaHistos[3]->Fill((*hSigma).first.second-1,(*hSigma).first.first);
	}
      }
      edm::LogWarning ("resolution") << "-------- wheel: "<<wheel.str()<<" sector: "<<sector.str()<<"  "<<theSigmaQReport->getMessage()<<" ------- "<<theSigmaQReport->getStatus();
    }
  }

}



void DTResolutionTest::endJob(){

  edm::LogVerbatim ("resolution") << "[DTResolutionTest] endjob called!";
  dbe->rmdir("DT/Tests/DTResolution");

}



string DTResolutionTest::getMEName(const DTSuperLayerId & slID) {
  
  stringstream wheel; wheel << slID.wheel();	
  stringstream station; station << slID.station();	
  stringstream sector; sector << slID.sector();	
  stringstream superLayer; superLayer << slID.superlayer();
  
  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderName = 
    folderRoot + "DT/DTResolutionAnalysisTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/";

  string histoTag = parameters.getUntrackedParameter<string>("histoTag", "hResDist");

  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str(); 
  
  return histoname;
  
}


void DTResolutionTest::bookHistos(const DTChamberId & ch) {

  stringstream wheel; wheel << ch.wheel();		
  stringstream sector; sector << ch.sector();	

  string MeanHistoName =  "MeanTest_W" + wheel.str() + "_Sec" + sector.str(); 
  string SigmaHistoName =  "SigmaTest_W" + wheel.str() + "_Sec" + sector.str(); 

  dbe->setCurrentFolder("DT/Tests/DTResolution");


  MeanHistos[make_pair(ch.wheel(),ch.sector())] = dbe->book1D(MeanHistoName.c_str(),MeanHistoName.c_str(),11,0,10);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(1,"MB1_SL1",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(2,"MB1_SL2",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(3,"MB1_SL3",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(4,"MB2_SL1",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(5,"MB2_SL2",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(6,"MB2_SL3",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(7,"MB3_SL1",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(8,"MB3_SL2",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(9,"MB3_SL3",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(10,"MB4_SL1",1);
  (MeanHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(11,"MB4_SL3",1);

  SigmaHistos[make_pair(ch.wheel(),ch.sector())] = dbe->book1D(SigmaHistoName.c_str(),SigmaHistoName.c_str(),11,0,10);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(1,"MB1_SL1",1);  
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(2,"MB1_SL2",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(3,"MB1_SL3",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(4,"MB2_SL1",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(5,"MB2_SL2",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(6,"MB2_SL3",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(7,"MB3_SL1",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(8,"MB3_SL2",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(9,"MB3_SL3",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(10,"MB4_SL1",1);
  (SigmaHistos[make_pair(ch.wheel(),ch.sector())])->setBinLabel(11,"MB4_SL3",1);


  string HistoName = "W" + wheel.str() + "_Sec" + sector.str(); 
  string MeanHistoNameSetRange = "MeanWrong_W" + wheel.str() + "_Sec" + sector.str() + "_SetRange";
  string SigmaHistoNameSetRange =  "SigmaWrong_W" + wheel.str() + "_Sec" + sector.str()  + "_SetRange";
  MeanHistosSetRange[HistoName] = dbe->book1D(MeanHistoNameSetRange.c_str(),MeanHistoNameSetRange.c_str(),10,0.5,10.5);
  SigmaHistosSetRange[HistoName] = dbe->book1D(SigmaHistoNameSetRange.c_str(),SigmaHistoNameSetRange.c_str(),10,0.5,10.5);
  string MeanHistoNameSetRange2D = "MeanWrong_W" + wheel.str() + "_Sec" + sector.str() + "_SetRange" + "_2D";
  string SigmaHistoNameSetRange2D =  "SigmaWrong_W" + wheel.str() + "_Sec" + sector.str()  + "_SetRange" + "_2D";
  MeanHistosSetRange2D[HistoName] = dbe->book2D(MeanHistoNameSetRange2D.c_str(),MeanHistoNameSetRange2D.c_str(),10, 0.5, 10.5, 100, -0.05, 0.05);
  SigmaHistosSetRange2D[HistoName] = dbe->book2D(SigmaHistoNameSetRange2D.c_str(),SigmaHistoNameSetRange2D.c_str(),10, 0.5, 10.5, 500, 0, 0.5);

}


void DTResolutionTest::bookHistos(int wh) {
  
  dbe->setCurrentFolder("DT/Tests/DTResolution/SummaryPlot");

  if(wheelMeanHistos.find(3) == wheelMeanHistos.end()){
    string histoName =  "MeanSummaryRes_testFailedByAtLeast%BadSL";
    wheelMeanHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,0,14,5,-2,2);
    wheelMeanHistos[3]->setBinLabel(1,"Sector1",1);
    wheelMeanHistos[3]->setBinLabel(1,"Sector1",1);
    wheelMeanHistos[3]->setBinLabel(2,"Sector2",1);
    wheelMeanHistos[3]->setBinLabel(3,"Sector3",1);
    wheelMeanHistos[3]->setBinLabel(4,"Sector4",1);
    wheelMeanHistos[3]->setBinLabel(5,"Sector5",1);
    wheelMeanHistos[3]->setBinLabel(6,"Sector6",1);
    wheelMeanHistos[3]->setBinLabel(7,"Sector7",1);
    wheelMeanHistos[3]->setBinLabel(8,"Sector8",1);
    wheelMeanHistos[3]->setBinLabel(9,"Sector9",1);
    wheelMeanHistos[3]->setBinLabel(10,"Sector10",1);
    wheelMeanHistos[3]->setBinLabel(11,"Sector11",1);
    wheelMeanHistos[3]->setBinLabel(12,"Sector12",1);
    wheelMeanHistos[3]->setBinLabel(13,"Sector13",1);
    wheelMeanHistos[3]->setBinLabel(14,"Sector14",1);
    wheelMeanHistos[3]->setBinLabel(1,"Wheel-2",2);
    wheelMeanHistos[3]->setBinLabel(2,"Wheel-1",2);
    wheelMeanHistos[3]->setBinLabel(3,"Wheel0",2);
    wheelMeanHistos[3]->setBinLabel(4,"Wheel+1",2);
    wheelMeanHistos[3]->setBinLabel(5,"Wheel+2",2);
  }

  if(wheelSigmaHistos.find(3) == wheelSigmaHistos.end()){
    string histoName =  "SigmaSummaryRes_testFailedByAtLeast%BadSL";
    wheelSigmaHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,0,14,5,-2,2);
    wheelSigmaHistos[3]->setBinLabel(1,"Sector1",1);
    wheelSigmaHistos[3]->setBinLabel(1,"Sector1",1);
    wheelSigmaHistos[3]->setBinLabel(2,"Sector2",1);
    wheelSigmaHistos[3]->setBinLabel(3,"Sector3",1);
    wheelSigmaHistos[3]->setBinLabel(4,"Sector4",1);
    wheelSigmaHistos[3]->setBinLabel(5,"Sector5",1);
    wheelSigmaHistos[3]->setBinLabel(6,"Sector6",1);
    wheelSigmaHistos[3]->setBinLabel(7,"Sector7",1);
    wheelSigmaHistos[3]->setBinLabel(8,"Sector8",1);
    wheelSigmaHistos[3]->setBinLabel(9,"Sector9",1);
    wheelSigmaHistos[3]->setBinLabel(10,"Sector10",1);
    wheelSigmaHistos[3]->setBinLabel(11,"Sector11",1);
    wheelSigmaHistos[3]->setBinLabel(12,"Sector12",1);
    wheelSigmaHistos[3]->setBinLabel(13,"Sector13",1);
    wheelSigmaHistos[3]->setBinLabel(14,"Sector14",1);
    wheelSigmaHistos[3]->setBinLabel(1,"Wheel-2",2);
    wheelSigmaHistos[3]->setBinLabel(2,"Wheel-1",2);
    wheelSigmaHistos[3]->setBinLabel(3,"Wheel0",2);
    wheelSigmaHistos[3]->setBinLabel(4,"Wheel+1",2);
    wheelSigmaHistos[3]->setBinLabel(5,"Wheel+2",2);
  }

  stringstream wheel; wheel <<wh;
  
  if(wheelMeanHistos.find(wh) == wheelMeanHistos.end()){
    string histoName =  "MeanSummaryRes_testFailed_W" + wheel.str();
    wheelMeanHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,0,14,11,0,11);
    wheelMeanHistos[wh]->setBinLabel(1,"Sector1",1);
    wheelMeanHistos[wh]->setBinLabel(2,"Sector2",1);
    wheelMeanHistos[wh]->setBinLabel(3,"Sector3",1);
    wheelMeanHistos[wh]->setBinLabel(4,"Sector4",1);
    wheelMeanHistos[wh]->setBinLabel(5,"Sector5",1);
    wheelMeanHistos[wh]->setBinLabel(6,"Sector6",1);
    wheelMeanHistos[wh]->setBinLabel(7,"Sector7",1);
    wheelMeanHistos[wh]->setBinLabel(8,"Sector8",1);
    wheelMeanHistos[wh]->setBinLabel(9,"Sector9",1);
    wheelMeanHistos[wh]->setBinLabel(10,"Sector10",1);
    wheelMeanHistos[wh]->setBinLabel(11,"Sector11",1);
    wheelMeanHistos[wh]->setBinLabel(12,"Sector12",1);
    wheelMeanHistos[wh]->setBinLabel(13,"Sector13",1);
    wheelMeanHistos[wh]->setBinLabel(14,"Sector14",1);
    wheelMeanHistos[wh]->setBinLabel(1,"MB1_SL1",2);
    wheelMeanHistos[wh]->setBinLabel(2,"MB1_SL2",2);
    wheelMeanHistos[wh]->setBinLabel(3,"MB1_SL3",2);
    wheelMeanHistos[wh]->setBinLabel(4,"MB2_SL1",2);
    wheelMeanHistos[wh]->setBinLabel(5,"MB2_SL2",2);
    wheelMeanHistos[wh]->setBinLabel(6,"MB2_SL3",2);
    wheelMeanHistos[wh]->setBinLabel(7,"MB3_SL1",2);
    wheelMeanHistos[wh]->setBinLabel(8,"MB3_SL2",2);
    wheelMeanHistos[wh]->setBinLabel(9,"MB3_SL3",2);
    wheelMeanHistos[wh]->setBinLabel(10,"MB4_SL1",2);
    wheelMeanHistos[wh]->setBinLabel(11,"MB4_SL3",2);
  }  

  if(wheelSigmaHistos.find(wh) == wheelSigmaHistos.end()){
    string histoName =  "SigmaSummaryRes_testFailed_W" + wheel.str();
    wheelSigmaHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,0,14,11,0,11);
    wheelSigmaHistos[wh]->setBinLabel(1,"Sector1",1);
    wheelSigmaHistos[wh]->setBinLabel(2,"Sector2",1);
    wheelSigmaHistos[wh]->setBinLabel(3,"Sector3",1);
    wheelSigmaHistos[wh]->setBinLabel(4,"Sector4",1);
    wheelSigmaHistos[wh]->setBinLabel(5,"Sector5",1);
    wheelSigmaHistos[wh]->setBinLabel(6,"Sector6",1);
    wheelSigmaHistos[wh]->setBinLabel(7,"Sector7",1);
    wheelSigmaHistos[wh]->setBinLabel(8,"Sector8",1);
    wheelSigmaHistos[wh]->setBinLabel(9,"Sector9",1);
    wheelSigmaHistos[wh]->setBinLabel(10,"Sector10",1);
    wheelSigmaHistos[wh]->setBinLabel(11,"Sector11",1);
    wheelSigmaHistos[wh]->setBinLabel(12,"Sector12",1);
    wheelSigmaHistos[wh]->setBinLabel(13,"Sector13",1);
    wheelSigmaHistos[wh]->setBinLabel(14,"Sector14",1);
    wheelSigmaHistos[wh]->setBinLabel(1,"MB1_SL1",2);
    wheelSigmaHistos[wh]->setBinLabel(2,"MB1_SL2",2);
    wheelSigmaHistos[wh]->setBinLabel(3,"MB1_SL3",2);
    wheelSigmaHistos[wh]->setBinLabel(4,"MB2_SL1",2);
    wheelSigmaHistos[wh]->setBinLabel(5,"MB2_SL2",2);
    wheelSigmaHistos[wh]->setBinLabel(6,"MB2_SL3",2);
    wheelSigmaHistos[wh]->setBinLabel(7,"MB3_SL1",2);
    wheelSigmaHistos[wh]->setBinLabel(8,"MB3_SL2",2);
    wheelSigmaHistos[wh]->setBinLabel(9,"MB3_SL3",2);
    wheelSigmaHistos[wh]->setBinLabel(10,"MB4_SL1",2);
    wheelSigmaHistos[wh]->setBinLabel(11,"MB4_SL3",2);
  }  
  
}
  
