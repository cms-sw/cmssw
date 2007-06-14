

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/05/22 07:06:21 $
 *  $Revision: 1.10 $
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

}


DTResolutionTest::~DTResolutionTest(){

  edm::LogVerbatim ("resolution") << "DTResolutionTest: analyzed " << nevents << " events";

}

void DTResolutionTest::endJob(){

  edm::LogVerbatim ("resolution") << "[DTResolutionTest] endjob called!";

  dbe->rmdir("DT/Tests/DTResolution");

}


void DTResolutionTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("resolution") <<"[DTResolutionTest]: BeginJob"; 

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTResolutionTest::bookHistos(const DTChamberId & ch) {

  stringstream wheel; wheel << ch.wheel();		
  stringstream sector; sector << ch.sector();	

  string MeanHistoName =  "MeanTest_W" + wheel.str() + "_Sec" + sector.str(); 
  string SigmaHistoName =  "SigmaTest_W" + wheel.str() + "_Sec" + sector.str(); 

  dbe->setCurrentFolder("DT/Tests/DTResolution");

  string HistoName = "W" + wheel.str() + "_Sec" + sector.str(); 

  MeanHistos[HistoName] = dbe->book1D(MeanHistoName.c_str(),MeanHistoName.c_str(),11,0,10);
  (MeanHistos[HistoName])->setBinLabel(1,"MB1_SL1",1);
  (MeanHistos[HistoName])->setBinLabel(2,"MB1_SL2",1);
  (MeanHistos[HistoName])->setBinLabel(3,"MB1_SL3",1);
  (MeanHistos[HistoName])->setBinLabel(4,"MB2_SL1",1);
  (MeanHistos[HistoName])->setBinLabel(5,"MB2_SL2",1);
  (MeanHistos[HistoName])->setBinLabel(6,"MB2_SL3",1);
  (MeanHistos[HistoName])->setBinLabel(7,"MB3_SL1",1);
  (MeanHistos[HistoName])->setBinLabel(8,"MB3_SL2",1);
  (MeanHistos[HistoName])->setBinLabel(9,"MB3_SL3",1);
  (MeanHistos[HistoName])->setBinLabel(10,"MB4_SL1",1);
  (MeanHistos[HistoName])->setBinLabel(11,"MB4_SL3",1);

  SigmaHistos[HistoName] = dbe->book1D(SigmaHistoName.c_str(),SigmaHistoName.c_str(),11,0,10);
  (SigmaHistos[HistoName])->setBinLabel(1,"MB1_SL1",1);  
  (SigmaHistos[HistoName])->setBinLabel(2,"MB1_SL2",1);
  (SigmaHistos[HistoName])->setBinLabel(3,"MB1_SL3",1);
  (SigmaHistos[HistoName])->setBinLabel(4,"MB2_SL1",1);
  (SigmaHistos[HistoName])->setBinLabel(5,"MB2_SL2",1);
  (SigmaHistos[HistoName])->setBinLabel(6,"MB2_SL3",1);
  (SigmaHistos[HistoName])->setBinLabel(7,"MB3_SL1",1);
  (SigmaHistos[HistoName])->setBinLabel(8,"MB3_SL2",1);
  (SigmaHistos[HistoName])->setBinLabel(9,"MB3_SL3",1);
  (SigmaHistos[HistoName])->setBinLabel(10,"MB4_SL1",1);
  (SigmaHistos[HistoName])->setBinLabel(11,"MB4_SL3",1);


  string MeanHistoNameSetRange = "MeanWrong_W" + wheel.str() + "_Sec" + sector.str() + "_SetRange";
  string SigmaHistoNameSetRange =  "SigmaWrong_W" + wheel.str() + "_Sec" + sector.str()  + "_SetRange";
  MeanHistosSetRange[HistoName] = dbe->book1D(MeanHistoNameSetRange.c_str(),MeanHistoNameSetRange.c_str(),10,0.5,10.5);
  SigmaHistosSetRange[HistoName] = dbe->book1D(SigmaHistoNameSetRange.c_str(),SigmaHistoNameSetRange.c_str(),10,0.5,10.5);
  string MeanHistoNameSetRange2D = "MeanWrong_W" + wheel.str() + "_Sec" + sector.str() + "_SetRange" + "_2D";
  string SigmaHistoNameSetRange2D =  "SigmaWrong_W" + wheel.str() + "_Sec" + sector.str()  + "_SetRange" + "_2D";
  MeanHistosSetRange2D[HistoName] = dbe->book2D(MeanHistoNameSetRange2D.c_str(),MeanHistoNameSetRange2D.c_str(),10, 0.5, 10.5, 100, -0.05, 0.05);
  SigmaHistosSetRange2D[HistoName] = dbe->book2D(SigmaHistoNameSetRange2D.c_str(),SigmaHistoNameSetRange2D.c_str(),10, 0.5, 10.5, 500, 0, 0.5);

}


void DTResolutionTest::analyze(const edm::Event& e, const edm::EventSetup& context){
  


  // counts number of updats (online mode) or number of events (standalone mode)
  nevents++;
  // if running in standalone perform diagnostic only after a reasonalbe amount of events
  if ( parameters.getUntrackedParameter<bool>("runningStandalone", false) && 
       nevents%parameters.getUntrackedParameter<int>("diagnosticPrescale", 1000) != 0 ) return;


  edm::LogVerbatim ("resolution") << "[DTResolutionTest]: "<<nevents<<" updates";

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
	if (MeanHistos.find(HistoName) == MeanHistos.end()) bookHistos((*ch_it)->id());
	MeanHistos.find(HistoName)->second->setBinContent(BinNumber, mean);	
	SigmaHistos.find(HistoName)->second->setBinContent(BinNumber, sigma);
      }
    }
  }

  // Mean test 
  cout<<"[DTResolutionTest]: Residuals Mean Tests results"<<endl;
  string MeanCriterionName = parameters.getUntrackedParameter<string>("meanTestName","ResidualsMeanInRange"); 
  for(map<string, MonitorElement*>::const_iterator hMean = MeanHistos.begin();
      hMean != MeanHistos.end();
      hMean++) {
    const QReport * theMeanQReport = (*hMean).second->getQReport(MeanCriterionName);
    if(theMeanQReport) {
      vector<dqm::me_util::Channel> badChannels = theMeanQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("resolution") << "Sector : "<<(*hMean).first<<" Bad mean channels: "<<(*channel).getBin()<<"  Contents : "<<(*channel).getContents();
	if (MeanHistosSetRange.find((*hMean).first) == MeanHistosSetRange.end()) bookHistos((*ch_it)->id());
	MeanHistosSetRange.find((*hMean).first)->second->Fill((*channel).getBin());
	if (MeanHistosSetRange2D.find((*hMean).first) == MeanHistosSetRange2D.end()) bookHistos((*ch_it)->id());
        MeanHistosSetRange2D.find((*hMean).first)->second->Fill((*channel).getBin(),(*channel).getContents());
      }
      edm::LogWarning ("resolution") << "-------- Sector : "<<(*hMean).first<<"  "<<theMeanQReport->getMessage()<<" ------- "<<theMeanQReport->getStatus(); 
    }
  }

  // Sigma test
  cout<<"[DTResolutionTest]: Residuals Sigma Tests results"<<endl;
  string SigmaCriterionName = parameters.getUntrackedParameter<string>("sigmaTestName","ResidualsSigmaInRange"); 
  for(map<string , MonitorElement*>::const_iterator hSigma = SigmaHistos.begin();
      hSigma != SigmaHistos.end();
      hSigma++) {
    const QReport * theSigmaQReport = (*hSigma).second->getQReport(SigmaCriterionName);
    if(theSigmaQReport) {
      vector<dqm::me_util::Channel> badChannels = theSigmaQReport->getBadChannels();
      for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	   channel != badChannels.end(); channel++) {
	edm::LogError ("resolution") << "Sector : "<<(*hSigma).first<<" Bad sigma channels: "<<(*channel).getBin()<<" Contents : "<<(*channel).getContents();
	if (SigmaHistosSetRange.find((*hSigma).first) == SigmaHistosSetRange.end()) bookHistos((*ch_it)->id());
        SigmaHistosSetRange.find((*hSigma).first)->second->Fill((*channel).getBin());
        if (SigmaHistosSetRange2D.find((*hSigma).first) == SigmaHistosSetRange2D.end()) bookHistos((*ch_it)->id());
        SigmaHistosSetRange2D.find((*hSigma).first)->second->Fill((*channel).getBin(),(*channel).getContents());
      }
      edm::LogWarning ("resolution") << "-------- Sector : "<<(*hSigma).first<<"  "<<theSigmaQReport->getMessage()<<" ------- "<<theSigmaQReport->getStatus();
    }
  }

  if (nevents%parameters.getUntrackedParameter<int>("resultsSavingRate",10) == 0){
    if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) 
      dbe->save(parameters.getUntrackedParameter<string>("outputFile", "DTResolutionTest.root"));
  }
}


string DTResolutionTest::getMEName(const DTSuperLayerId & slID) {
  
  stringstream wheel; wheel << slID.wheel();	
  stringstream station; station << slID.station();	
  stringstream sector; sector << slID.sector();	
  stringstream superLayer; superLayer << slID.superlayer();
  
  string folderName = 
    "Collector/FU0/DT/DTResolutionAnalysisTask/Wheel" +  wheel.str() +
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
