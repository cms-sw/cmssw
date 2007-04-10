

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/03/30 16:10:06 $
 *  $Revision: 1.6 $
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

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;


DTResolutionTest::DTResolutionTest(const edm::ParameterSet& ps){

  debug = ps.getUntrackedParameter<bool>("debug", "false");
  if(debug)
    cout<<"[DTResolutionTest]: Constructor"<<endl;

  parameters = ps;

  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);

}


DTResolutionTest::~DTResolutionTest(){

  if(debug)
    cout << "DTResolutionTest: analyzed " << nevents << " events" << endl;

}

void DTResolutionTest::endJob(){

  if(debug)
    cout<<"[DTResolutionTest] endjob called!"<<endl;

  dbe->rmdir("DT/Tests/DTResolution");

}


void DTResolutionTest::beginJob(const edm::EventSetup& context){

  if(debug)
    cout<<"[DTResolutionTest]: BeginJob"<<endl;

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
}


void DTResolutionTest::analyze(const edm::Event& e, const edm::EventSetup& context){
  
  nevents++;
  if (nevents%1 == 0 && debug) 
    cout<<"[DTResolutionTest]: "<<nevents<<" updates"<<endl;

  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

  cout<<endl;
  cout<<"[DTResolutionTest]: Residual Distribution tests results"<<endl;
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

      stringstream wheel; wheel << chID.wheel();
      stringstream sector; sector << chID.sector();

      string HistoName = "W" + wheel.str() + "_Sec" + sector.str(); 

      MonitorElement * res_histo = dbe->get(getMEName(slID));
      if (res_histo) {
	// gaussian test
	string GaussianCriterionName = 
	  parameters.getUntrackedParameter<string>("resDistributionTestName",
						   "ResidualsDistributionGaussianTest");
	const QReport * GaussianReport = res_histo->getQReport(GaussianCriterionName);
	if(GaussianReport){
	  cout<<"-------- "<<GaussianReport->getMessage()<<" ------- "<<GaussianReport->getStatus()<<endl;
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
	cout<<"Bad mean channels: "<<(*channel).getBin()<<" "<<(*channel).getContents()<<endl;
      }
      cout<<"-------- "<<theMeanQReport->getMessage()<<" ------- "<<theMeanQReport->getStatus()<<endl;
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
	cout<<"Bad sigma channels: "<<(*channel).getBin()<<" "<<(*channel).getContents()<<endl;
      }
      cout<<"-------- "<<theSigmaQReport->getMessage()<<" ------- "<<theSigmaQReport->getStatus()<<endl;
    }
  }

  if (nevents%parameters.getUntrackedParameter<int>("resultsSavingRate",100) == 0){
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
