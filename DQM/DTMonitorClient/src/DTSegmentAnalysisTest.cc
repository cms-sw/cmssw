/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 */


#include <DQM/DTMonitorClient/src/DTSegmentAnalysisTest.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include "DataFormats/Common/interface/Handle.h" 
#include <FWCore/Framework/interface/ESHandle.h>
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

#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;


DTSegmentAnalysisTest::DTSegmentAnalysisTest(const ParameterSet& ps){

  LogTrace ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest") << "[DTSegmentAnalysisTest]: Constructor";
  parameters = ps;

  // get the cfi parameters
  detailedAnalysis = parameters.getUntrackedParameter<bool>("detailedAnalysis",false);
  normalizeHistoPlots  = parameters.getUntrackedParameter<bool>("normalizeHistoPlots",false);
  runOnline  = parameters.getUntrackedParameter<bool>("runOnline",true);
  // top folder for the histograms in DQMStore
  topHistoFolder = ps.getUntrackedParameter<string>("topHistoFolder","DT/02-Segments");
  // hlt DQM mode

  hltDQMMode = ps.getUntrackedParameter<bool>("hltDQMMode",false);
  nMinEvts  = ps.getUntrackedParameter<int>("nEventsCert", 5000);
  maxPhiHit  = ps.getUntrackedParameter<int>("maxPhiHit", 7);
  maxPhiZHit  = ps.getUntrackedParameter<int>("maxPhiZHit", 11);

  nevents = 0;

  bookingdone = 0;

}


DTSegmentAnalysisTest::~DTSegmentAnalysisTest(){

  LogTrace ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest") << "DTSegmentAnalysisTest: analyzed " << nevents << " events";
}

void DTSegmentAnalysisTest::beginRun(const Run& run, const EventSetup& context){

  LogTrace ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest") <<"[DTSegmentAnalysisTest]: BeginRun"; 

  context.get<MuonGeometryRecord>().get(muonGeom);
}


void DTSegmentAnalysisTest::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, 
                                                  edm::LuminosityBlock const & lumiSeg, edm::EventSetup const & context) {

  // book the histos

  if (!bookingdone) bookHistos(ibooker);  
  bookingdone = 1; 

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
 
  if (runOnline) {
    LogTrace ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest")
      <<"[DTSegmentAnalysisTest]: End of LS " << nLumiSegs 
      << ". Client called in online mode , perform DQM client operation";

    performClientDiagnostic(igetter);
  }

}

void DTSegmentAnalysisTest::endRun(Run const& run, EventSetup const& context) {
}

void DTSegmentAnalysisTest::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  if (!runOnline) {

    LogTrace ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest")
      <<"[DTSegmentAnalysisTest]: endJob. Client called in offline mode , perform DQM client operation";

    performClientDiagnostic(igetter);
  }

  if(normalizeHistoPlots) {
    LogTrace ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest") << " Performing time-histo normalization" << endl;
    MonitorElement* hNevtPerLS = 0;

    if(hltDQMMode) hNevtPerLS = igetter.get(topHistoFolder + "/NevtPerLS");
    else  hNevtPerLS = igetter.get("DT/EventInfo/NevtPerLS");

    if(hNevtPerLS != 0) {
      for(int wheel = -2; wheel != 3; ++wheel) { // loop over wheels
	for(int sector = 1; sector <= 12; ++sector) { // loop over sectors
	  stringstream wheelstr; wheelstr << wheel;	
	  stringstream sectorstr; sectorstr << sector;
	  string sectorHistoName = topHistoFolder + "/Wheel" + wheelstr.str() +
	    "/Sector" + sectorstr.str() +
	    "/NSegmPerEvent_W" + wheelstr.str() +
	    "_Sec" + sectorstr.str();

	  //FR get the histo from here (igetter available!) ...
          MonitorElement* histoGot=igetter.get(sectorHistoName);

          //FR ...and just make with it a DTTimeEvolutionHisto
          DTTimeEvolutionHisto hNSegmPerLS(histoGot);

	  hNSegmPerLS.normalizeTo(hNevtPerLS);
	}
      }
    } else {
      LogError ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest") << "Histo NevtPerLS not found!" << endl;
    }
  }
}

void DTSegmentAnalysisTest::performClientDiagnostic(DQMStore::IGetter & igetter) {

  summaryHistos[3]->Reset();
  summaryHistos[4]->Reset();
  vector<const DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<const DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();
 
  for (; ch_it != ch_end; ++ch_it) {
    DTChamberId chID = (*ch_it)->id();
    
    MonitorElement * hNHits = igetter.get(getMEName(chID, "h4DSegmNHits"));
    MonitorElement * hSegmOcc = igetter.get(getMEName(chID, "numberOfSegments"));
   
    if (hNHits && hSegmOcc) {
      
      TH1F * hNHits_root = hNHits->getTH1F();
      TH2F * hSegmOcc_root = hSegmOcc->getTH2F();
      TH2F * summary_histo_root = summaryHistos[3]->getTH2F();
      
      int sector = chID.sector();
      if(sector == 13) sector=4;
      if(sector == 14) sector=10;
      
      
      if((chID.station()!=4 && hNHits_root->GetMaximumBin() < maxPhiZHit)||
	 (chID.station()==4 &&  hNHits_root->GetMaximumBin() < maxPhiHit)){
	summaryHistos[chID.wheel()]->setBinContent(sector, chID.station(),1);
	if(summary_histo_root->GetBinContent(sector, chID.wheel()+3)<1)
	  summaryHistos[3]->setBinContent(sector, chID.wheel()+3,1);  
      }
      else
	summaryHistos[chID.wheel()]->setBinContent(sector, chID.station(),0);
    
      if(detailedAnalysis) {
	if(chID.station()!=4)
	  segmRecHitHistos[make_pair(chID.wheel(),chID.sector())]->Fill(chID.station(),abs(12-hNHits_root->GetMaximumBin()));
	else
	   segmRecHitHistos[make_pair(chID.wheel(),chID.sector())]->Fill(chID.station(),abs(8-hNHits_root->GetMaximumBin()));
      }

      TH2F * summary2_histo_root = summaryHistos[3]->getTH2F();
      
      if(hSegmOcc_root->GetBinContent(sector,chID.station())==0){
	summaryHistos[chID.wheel()]->setBinContent(sector, chID.station(),2);
	if(summary2_histo_root->GetBinContent(sector, chID.wheel()+3)<2)
	  summaryHistos[3]->setBinContent(sector, chID.wheel()+3,2);
      } else {
	// Fill the percentage of segment occupancy
	float weight = 1./4.;
	if((sector == 4 || sector == 10) && chID.station() == 4) weight = 1./8.;
	summaryHistos[4]->Fill(sector, chID.wheel(),weight);
      }
      
    } else {
      LogVerbatim ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest")
	<< "[DTSegmentAnalysisTest]: histos not found!!"; // FIXME
    }

    if(detailedAnalysis){ // switch on detailed analysis
   
      //test on chi2 segment quality

      MonitorElement * chi2_histo = igetter.get(getMEName(chID, "h4DChi2"));
      if(chi2_histo) {
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
      
	if(chi2_histo_root->GetEntries()!=0){
	  double badSegmentsPercentual= badSegments/double(chi2_histo_root->GetEntries());
	  chi2Histos[make_pair(chID.wheel(),chID.sector())]->Fill(chID.station(),badSegmentsPercentual);
	}
      } else {
	LogVerbatim ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest")
	  <<"[DTSegmentAnalysisTest]: Histo: " << getMEName(chID, "h4DChi2") << " not found!" << endl;
      }
    } // end of switch for detailed analysis
    
  } //loop over all the chambers

  string nEvtsName = "DT/EventInfo/Counters/nProcessedEventsSegment";

  MonitorElement * meProcEvts = igetter.get(nEvtsName);

  if (meProcEvts) {
    int nProcEvts = meProcEvts->getFloatValue();
    summaryHistos[4]->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
  } else {
    summaryHistos[4]->setEntries(nMinEvts + 1);
    LogVerbatim ("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest] ME: "
		       <<  nEvtsName << " not found!" << endl;
  }

  if(detailedAnalysis){
    
    string chi2CriterionName = parameters.getUntrackedParameter<string>("chi2TestName","chi2InRange");
    for(map<pair<int, int>, MonitorElement*> ::const_iterator histo = chi2Histos.begin();
	histo != chi2Histos.end();
	histo++) {

      const QReport * theChi2QReport = (*histo).second->getQReport(chi2CriterionName);
      if(theChi2QReport) {
	vector<dqm::me_util::Channel> badChannels = theChi2QReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	     channel != badChannels.end(); channel++) {

	  LogError ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest") << "Wheel: "<<(*histo).first.first
								   << " Sector: "<<(*histo).first.second
								   << " Bad stations: "<<(*channel).getBin()
								   <<"  Contents : "<<(*channel).getContents();
	}
      }
    }

 
    string segmRecHitCriterionName = parameters.getUntrackedParameter<string>("segmRecHitTestName","segmRecHitInRange");
    for(map<pair<int, int>, MonitorElement*> ::const_iterator histo = segmRecHitHistos.begin();
	histo != segmRecHitHistos.end();
	histo++) {

      const QReport * theSegmRecHitQReport = (*histo).second->getQReport(segmRecHitCriterionName);
      if(theSegmRecHitQReport) {
	vector<dqm::me_util::Channel> badChannels = theSegmRecHitQReport->getBadChannels();
	for (vector<dqm::me_util::Channel>::iterator channel = badChannels.begin(); 
	     channel != badChannels.end(); channel++) {

	  LogError ("DTDQM|DTMonitorClient|DTSegmentAnalysisTest") << "Wheel: "<<(*histo).first.first
								   << " Sector: "<<(*histo).first.second
								   << " Bad stations on recHit number: "
								   <<(*channel).getBin()
								   <<"  Contents : "
								   <<(*channel).getContents();
	}
      }
    }

  } // end of detailedAnalysis
}


string DTSegmentAnalysisTest::getMEName(const DTChamberId & chID, string histoTag) {
  
  stringstream wheel; wheel << chID.wheel();	
  stringstream station; station << chID.station();	
  stringstream sector; sector << chID.sector();	
  
  string folderName = 
    topHistoFolder + "/Wheel" +  wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/";

  string histoname = folderName + histoTag  
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str(); 
  
  if(histoTag == "numberOfSegments")
    histoname = 
      topHistoFolder + "/Wheel" +  wheel.str() + "/" +
      histoTag  + + "_W" + wheel.str();

  return histoname;
  
}

void DTSegmentAnalysisTest::bookHistos(DQMStore::IBooker & ibooker) {

  for(int wh=-2; wh<=2; wh++){
      stringstream wheel; wheel << wh;
      string histoName =  "segmentSummary_W" + wheel.str();

      ibooker.setCurrentFolder(topHistoFolder);

      summaryHistos[wh] = ibooker.book2D(histoName.c_str(),histoName.c_str(),12,1,13,4,1,5);
      summaryHistos[wh]->setAxisTitle("Sector",1);
      summaryHistos[wh]->setBinLabel(1,"MB1",2);
      summaryHistos[wh]->setBinLabel(2,"MB2",2);
      summaryHistos[wh]->setBinLabel(3,"MB3",2);
      summaryHistos[wh]->setBinLabel(4,"MB4",2);

      if(detailedAnalysis){
	for(int sect=1; sect<=14; sect++){
	  stringstream sector; sector << sect;
	  string chi2HistoName =  "chi2BadSegmPercentual_W" + wheel.str() + "_Sec" + sector.str();
	  ibooker.setCurrentFolder(topHistoFolder + "/Wheel" + wheel.str() + "/Tests");
	  chi2Histos[make_pair(wh,sect)] = ibooker.book1D(chi2HistoName.c_str(),chi2HistoName.c_str(),4,1,5);
	  chi2Histos[make_pair(wh,sect)]->setBinLabel(1,"MB1");
	  chi2Histos[make_pair(wh,sect)]->setBinLabel(2,"MB2");
	  chi2Histos[make_pair(wh,sect)]->setBinLabel(3,"MB3");
	  chi2Histos[make_pair(wh,sect)]->setBinLabel(4,"MB4");
	  
	  string segmHistoName =  "residualsOnSegmRecHitNumber_W" + wheel.str() + "_Sec" + sector.str();
	  segmRecHitHistos[make_pair(wh,sect)] = ibooker.book1D(segmHistoName.c_str(),segmHistoName.c_str(),4,1,5);
	  segmRecHitHistos[make_pair(wh,sect)]->setBinLabel(1,"MB1");
	  segmRecHitHistos[make_pair(wh,sect)]->setBinLabel(2,"MB2");
	  segmRecHitHistos[make_pair(wh,sect)]->setBinLabel(3,"MB3");
	  segmRecHitHistos[make_pair(wh,sect)]->setBinLabel(4,"MB4");
	  
	}
      }
  }
  
  string histoName =  "segmentSummary";

  ibooker.setCurrentFolder(topHistoFolder);

  summaryHistos[3] = ibooker.book2D(histoName.c_str(),histoName.c_str(),12,1,13,5,-2,3);
  summaryHistos[3]->setAxisTitle("Sector",1);
  summaryHistos[3]->setAxisTitle("Wheel",2); 

  summaryHistos[4] = ibooker.book2D("SegmentGlbSummary",histoName.c_str(),12,1,13,5,-2,3);
  summaryHistos[4]->setAxisTitle("Sector",1);
  summaryHistos[4]->setAxisTitle("Wheel",2); 


}
  

  




