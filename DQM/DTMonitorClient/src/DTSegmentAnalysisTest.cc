

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/04/23 15:02:10 $
 *  $Revision: 1.4 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTSegmentAnalysisTest.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h" 
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;


DTSegmentAnalysisTest::DTSegmentAnalysisTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("segment") << "[DTSegmentAnalysisTest]: Constructor";
  parameters = ps;

  dbe = edm::Service<DQMStore>().operator->();

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);
  badChpercentual = parameters.getUntrackedParameter<int>("badChpercentual", 10);

}


DTSegmentAnalysisTest::~DTSegmentAnalysisTest(){

  edm::LogVerbatim ("segment") << "DTSegmentAnalysisTest: analyzed " << nevents << " events";
}


void DTSegmentAnalysisTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("segment") <<"[DTSegmentAnalysisTest]: BeginJob"; 

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTSegmentAnalysisTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("segment") <<"[DTSegmentAnalysisTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}


void DTSegmentAnalysisTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("segment") << "[DTSegmentAnalysisTest]: "<<nevents<<" events";

}


void DTSegmentAnalysisTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  
  edm::LogVerbatim ("segment") <<"[DTSegmentAnalysisTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();

  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  for(map<int, MonitorElement*> ::const_iterator histo = wheelHistos.begin();
      histo != wheelHistos.end();
      histo++) {
    (*histo).second->Reset();
  }
  map <pair<int,int>, int> cmsHistos;
  cmsHistos.clear();
  map <pair<int,int>, bool> filled;
  for(int i=-2; i<3; i++){
    for(int j=1; j<15; j++){
      filled[make_pair(i,j)]=false;
    }
  }


  edm::LogVerbatim ("segment") <<"[DTSegmentAnalysisTest]: "<<nLumiSegs<<" updates";

  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();
  
  for (; ch_it != ch_end; ++ch_it) {
    
    DTChamberId chID = (*ch_it)->id();
    
    //test on chi2 segment quality
    MonitorElement * chi2_histo = dbe->get(getMEName(chID, "h4DChi2"));
    if (chi2_histo) {

      edm::LogVerbatim ("segment") <<"[DTSegmentAnalysisTest]: I've got the histo of the segment chi2!!";

      TH1F * chi2_histo_root = chi2_histo->getTH1F();
      double threshold = parameters.getUntrackedParameter<double>("chi2Threshold", 5);
      double maximum = chi2_histo_root->GetXaxis()->GetXmax();
      double minimum = chi2_histo_root->GetXaxis()->GetXmin();
      int nbins = chi2_histo_root->GetXaxis()->GetNbins();
      int thresholdBin = int(threshold/((maximum-minimum)/nbins));
      
      int badSegments=0;
      for(int bin=thresholdBin; bin<=nbins; bin++){
	badSegments+=chi2_histo_root->GetBinContent(bin);
      }
   
      int badSegmPercentual = parameters.getUntrackedParameter<int>("badSegmPercentual", 30);
      if(double(badSegments)/chi2_histo_root->GetEntries()>double(badSegmPercentual)/100){
	if(wheelHistos.find(chID.wheel()) == wheelHistos.end()) bookHistos(chID.wheel());
	wheelHistos[chID.wheel()]->Fill(chID.sector()-1,chID.station()-1);
	cmsHistos[make_pair(chID.wheel(),chID.sector())]++;
	if((chID.sector()<13 &&
	    double(cmsHistos[make_pair(chID.wheel(),chID.sector())])/4>double(badChpercentual)/100 &&
	    filled[make_pair(chID.wheel(),chID.sector())]==false) ||
	   (chID.sector()>=13 && 
	    filled[make_pair(chID.wheel(),chID.sector())]==false)){
	  filled[make_pair(chID.wheel(),chID.sector())]=true;
	  wheelHistos[3]->Fill(chID.sector()-1,chID.wheel());
	}
      }

    }

    //summary of the number of hits per segment
    MonitorElement * nhit_histo = dbe->get(getMEName(chID, "h4DSegmNHits"));
    if (nhit_histo) {
      
      edm::LogVerbatim ("segment") <<"[DTSegmentAnalysisTest]: I've got the histo with the number of hit per segment!!";  

      TH1F * nhit_histo_root = nhit_histo->getTH1F();
      if(wheelHistos.find(chID.wheel()+6) == wheelHistos.end()) bookHistos(chID.wheel());
      for(int bin=0; bin<=nhit_histo_root->GetXaxis()->GetNbins(); bin++){
	wheelHistos[chID.wheel()+6]->Fill(chID.sector()-1,bin-1,nhit_histo_root->GetBinContent(bin));
      }

    }
	  
  } //loop over all the chambers

}



string DTSegmentAnalysisTest::getMEName(const DTChamberId & chID, string histoTag) {
  
  stringstream wheel; wheel << chID.wheel();	
  stringstream station; station << chID.station();	
  stringstream sector; sector << chID.sector();	
  
  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderName = 
    folderRoot + "DT/DTSegmentAnalysisTask/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/";

  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 
  
  return histoname;
  
}


void DTSegmentAnalysisTest::bookHistos(int wh) {
  
  dbe->setCurrentFolder("DT/Tests/DTSegmentAnalysisTest/SummaryPlot");

  if(wheelHistos.find(3) == wheelHistos.end()){
    string histoName =  "chi2Summary_testFailedByAtLeastBadCH";
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
  string histoName =  "chi2Summary_testFailed_W" + wheel.str();
  wheelHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,0,14,4,0,4);
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
  wheelHistos[wh]->setBinLabel(1,"MB1",2);
  wheelHistos[wh]->setBinLabel(2,"MB2",2);
  wheelHistos[wh]->setBinLabel(3,"MB3",2);
  wheelHistos[wh]->setBinLabel(4,"MB4",2);

  histoName =  "NumberOfHitsPerSegm_W" + wheel.str();
  wheelHistos[wh+6] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,0,14,20,0,20);
  wheelHistos[wh+6]->setBinLabel(1,"Sector1",1);
  wheelHistos[wh+6]->setBinLabel(2,"Sector2",1);
  wheelHistos[wh+6]->setBinLabel(3,"Sector3",1);
  wheelHistos[wh+6]->setBinLabel(4,"Sector4",1);
  wheelHistos[wh+6]->setBinLabel(5,"Sector5",1);
  wheelHistos[wh+6]->setBinLabel(6,"Sector6",1);
  wheelHistos[wh+6]->setBinLabel(7,"Sector7",1);
  wheelHistos[wh+6]->setBinLabel(8,"Sector8",1);
  wheelHistos[wh+6]->setBinLabel(9,"Sector9",1);
  wheelHistos[wh+6]->setBinLabel(10,"Sector10",1);
  wheelHistos[wh+6]->setBinLabel(11,"Sector11",1);
  wheelHistos[wh+6]->setBinLabel(12,"Sector12",1);
  wheelHistos[wh+6]->setBinLabel(13,"Sector13",1);
  wheelHistos[wh+6]->setBinLabel(14,"Sector14",1);

}
  

  

