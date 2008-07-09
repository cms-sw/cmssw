

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/06/03 16:40:43 $
 *  $Revision: 1.11 $
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

  // get the cfi parameters
  detailedAnalysis = parameters.getUntrackedParameter<bool>("detailedAnalysis","false");
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

  // book the histos
  bookHistos();  
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


  // for detailed anlysis
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
  // end of detailed analysis

  edm::LogVerbatim ("segment") <<"[DTSegmentAnalysisTest]: "<<nLumiSegs<<" updates";

  summaryHistos[3]->Reset();
  vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();
  
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId chID = (*ch_it)->id();
    
    MonitorElement * segm_histo = dbe->get(getMEName(chID, "h4DSegmNHits"));
    MonitorElement * summary_histo = dbe->get(getMEName(chID, "numberOfSegments"));

    if (segm_histo && summary_histo){
	edm::LogVerbatim ("segment") <<"[DTSegmentAnalysisTest]: I've got the recHits histo and the summary!!";

	TH1F * segmHit_histo_root = segm_histo->getTH1F();
	TH2F * segm_histo_root = summary_histo->getTH2F();
	TH2F * summary_histo_root = summaryHistos[3]->getTH2F();

	int sector = chID.sector();
	if(sector == 13) sector=4;
	if(sector == 14) sector=10;


	if((chID.station()!=4 && segmHit_histo_root->GetMaximumBin() != 12)||
	   (chID.station()==4 &&  segmHit_histo_root->GetMaximumBin() != 8)){
	  summaryHistos[chID.wheel()]->setBinContent(sector, chID.station(),1);
	  if(summary_histo_root->GetBinContent(sector, chID.wheel()+3)<1)
	    summaryHistos[3]->setBinContent(sector, chID.wheel()+3,1);
	}
	else
	  summaryHistos[chID.wheel()]->setBinContent(sector, chID.station(),0);

	TH2F * summary2_histo_root = summaryHistos[3]->getTH2F();

	if(segm_histo_root->GetBinContent(sector,chID.station())==0){
	  summaryHistos[chID.wheel()]->setBinContent(sector, chID.station(),2);
	  if(summary2_histo_root->GetBinContent(sector, chID.wheel()+3)<2)
	    summaryHistos[3]->setBinContent(sector, chID.wheel()+3,2);
	}

      }


    if(detailedAnalysis){ // switch on detailed analysis

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
	
	double badSegments=0;
	for(int bin=thresholdBin; bin<=nbins; bin++){
	  badSegments+=chi2_histo_root->GetBinContent(bin);
	}
	
	int badSegmPercentual = parameters.getUntrackedParameter<int>("badSegmPercentual", 30);
	if(double(badSegments)/chi2_histo_root->GetEntries()>double(badSegmPercentual)/100){
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
	for(int bin=0; bin<=nhit_histo_root->GetXaxis()->GetNbins(); bin++){
	  wheelHistos[chID.wheel()+6]->Fill(chID.sector()-1,bin-1,nhit_histo_root->GetBinContent(bin));
	}
      }
	
    } // end of switch for detailed analysis
    
    
    
  } //loop over all the chambers


}



string DTSegmentAnalysisTest::getMEName(const DTChamberId & chID, string histoTag) {
  
  stringstream wheel; wheel << chID.wheel();	
  stringstream station; station << chID.station();	
  stringstream sector; sector << chID.sector();	
  
  string folderRoot = parameters.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");
  string folderName = 
    folderRoot + "DT/Segments/Wheel" +  wheel.str() +
    "/Station" + station.str() +
    "/Sector" + sector.str() + "/";

  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 
  
  if(histoTag == "numberOfSegments")
    histoname = 
      folderRoot + "DT/Segments/Wheel" +  wheel.str() + "/" +
      histoTag  + + "_W" + wheel.str();

  return histoname;
  
}


void DTSegmentAnalysisTest::bookHistos() {
  
  dbe->setCurrentFolder("DT/Segments");

  for(int wh=-2; wh<=2; wh++){
      stringstream wheel; wheel << wh;
      string histoName =  "segmentSummary_W" + wheel.str();
      summaryHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),12,1,13,4,1,5);
      summaryHistos[wh]->setAxisTitle("Sector",1);
      summaryHistos[wh]->setBinLabel(1,"MB1",2);
      summaryHistos[wh]->setBinLabel(2,"MB2",2);
      summaryHistos[wh]->setBinLabel(3,"MB3",2);
      summaryHistos[wh]->setBinLabel(4,"MB4",2);  
  }
  

  string histoName =  "segmentSummary";
    summaryHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),12,1,13,5,-2,3);
    summaryHistos[3]->setAxisTitle("Sector",1);
    summaryHistos[3]->setAxisTitle("Wheel",2);
 


  if(detailedAnalysis){ // switch on detailed analysis

    for(int wh=-2; wh<=2; wh++){
      stringstream wheel; wheel << wh;
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

  } // // end of switch for detailed analysis


}
  

  

