

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/18 15:51:42 $
 *  $Revision: 1.12 $
 *  \author G. Mila - INFN Torino
 */


#include <DQM/DTMonitorClient/src/DTResolutionAnalysisTest.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <sstream>
#include <math.h>


using namespace edm;
using namespace std;


DTResolutionAnalysisTest::DTResolutionAnalysisTest(const edm::ParameterSet& ps){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "[DTResolutionAnalysisTest]: Constructor";

  dbe = edm::Service<DQMStore>().operator->();

  prescaleFactor = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);
  folderRoot = ps.getUntrackedParameter<string>("folderRoot", "Collector/FU0/");

  // permitted test range
  permittedMeanRange = ps.getUntrackedParameter<double>("permittedMeanRange",0.005); 
  permittedSigmaRange = ps.getUntrackedParameter<double>("permittedSigmaRange",0.01); 
  
 }


DTResolutionAnalysisTest::~DTResolutionAnalysisTest(){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "DTResolutionAnalysisTest: analyzed " << nevents << " events";

}


void DTResolutionAnalysisTest::beginJob(const edm::EventSetup& context){

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") <<"[DTResolutionAnalysisTest]: BeginJob"; 

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

  // loop over all the CMS wheels, sectors & book the summary histos
  for (int wheel=-2; wheel<=2; wheel++){
    bookHistos(wheel);
    for (int sector=1; sector<=12; sector++){
       bookHistos(wheel, sector);
    }
  }

}


void DTResolutionAnalysisTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") <<"[DTResolutionAnalysisTest]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

  // Reset the summary histo
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

}


void DTResolutionAnalysisTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;

}



void DTResolutionAnalysisTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") <<"[DTResolutionAnalysisTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
  
  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

   vector<DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator ch_end = muonGeom->chambers().end();

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
      MonitorElement * res_histo = dbe->get(getMEName(slID));

      if(res_histo){

	// Gaussian Fit
	float statMean = res_histo->getMean(1);
	float statSigma = res_histo->getRMS(1);
	Double_t mean = -1;
	Double_t sigma = -1;
	TH1F * histo_root = res_histo->getTH1F();
	if(histo_root->GetEntries()>20){
	  TF1 *gfit = new TF1("Gaussian","gaus",(statMean-(2*statSigma)),(statMean+(2*statSigma)));
	  try {
	    histo_root->Fit(gfit, "Q0");
	  } catch (...) {
	    edm::LogWarning ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask")
	      << "[DTResolutionAnalysisTask]: Exception when fitting SL : " << slID;
	    continue;
	  }
	  if(gfit){
	    mean = gfit->GetParameter(1); 
	    sigma = gfit->GetParameter(2);
	  }
	  delete gfit;
	}
	else{
	  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask")
	    << "[DTResolutionAnalysisTask] Fit of " << slID
	    << " not performed because # entries < 20 ";
	}
	
	int BinNumber = entry+slID.superLayer();
	if(BinNumber == 12) BinNumber=11;
	if((slID.sector()==13 || slID.sector()==14)  && slID.superLayer()==1) BinNumber=12;
	if((slID.sector()==13 || slID.sector()==14) && slID.superLayer()==3) BinNumber=13;

	// Fill the summary histos
	if(slID.sector()<13){
	  MeanHistos[make_pair(slID.wheel(),slID.sector())]->setBinContent(BinNumber, mean);	
	  SigmaHistos[make_pair(slID.wheel(),slID.sector())]->setBinContent(BinNumber, sigma);
	}
	if(slID.sector()==13){
	  MeanHistos[make_pair(slID.wheel(),4)]->setBinContent(BinNumber, mean);	
	  SigmaHistos[make_pair(slID.wheel(),4)]->setBinContent(BinNumber, sigma);
	}
	if(slID.sector()==14){
	  MeanHistos[make_pair(slID.wheel(),10)]->setBinContent(BinNumber, mean);	
	  SigmaHistos[make_pair(slID.wheel(),10)]->setBinContent(BinNumber, sigma);
	}
      }

    } // loop on SLs
  } // Loop on Stations
  

  for(int wheel=-2; wheel<=2; wheel++){
    for(int sector=1; sector<=12; sector++){

      int lastBin=-1;
      if(sector!=4 && sector!=10) lastBin=11;
      else lastBin=13;

      for (int bin=1; bin<=lastBin; bin++){

	// Mean test
	double mean = MeanHistos.find(make_pair(wheel,sector))->second->getBinContent(bin);
	if(mean<(-permittedMeanRange) || mean>permittedMeanRange){
// 	  edm::LogError("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "Bad mean channel: wh: " << wheel
// 									  << " st: " << stationFromBin(bin)
// 									  << " sect: " <<sector
// 									  << " sl: " << slFromBin(bin)
// 									  << " mean (cm): " << mean;
	  // fill the wheel summary histos
	  if(bin<12){
	    wheelMeanHistos[wheel]->Fill(sector,bin);
	    wheelMeanHistos[3]->Fill(sector,wheel);
	  }
	  else{
	    wheelMeanHistos[wheel]->Fill(sector,11);
	    wheelMeanHistos[3]->Fill(sector,wheel);
	  }
	}
	
	// Sigma test
	double sigma = SigmaHistos.find(make_pair(wheel,sector))->second->getBinContent(bin);
	if(sigma>permittedSigmaRange){
// 	  edm::LogError("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "Bad sigma: wh: " << wheel
// 									  << " st: " << stationFromBin(bin)
// 									  << " sect: " <<sector
// 									  << " sl: " << slFromBin(bin)
// 									  << " sigma (cm): " << sigma;
	  // fill the wheel summary histos
	  if(bin<12){
	    wheelSigmaHistos[wheel]->Fill(sector,bin);
	    wheelSigmaHistos[3]->Fill(sector,wheel);
	  }
	  else{
	    wheelSigmaHistos[wheel]->Fill(sector,11);
	    wheelSigmaHistos[3]->Fill(sector,wheel);
	  }
	}

      } // loop over bins

    } // loop over sectors
  } // loop over wheels

}



void DTResolutionAnalysisTest::bookHistos(int wh) { 

  stringstream wheel; wheel <<wh;

  dbe->setCurrentFolder("DT/02-Segments/00-MeanRes");
  string histoName =  "MeanSummaryRes_W" + wheel.str();
  stringstream meanRange; meanRange << (permittedMeanRange*10000);
  string histoTitle = "# of SL with |mean of res.| > " + meanRange.str() + "#mum (Wheel " + wheel.str() + ")";
  wheelMeanHistos[wh] = dbe->book2D(histoName.c_str(),histoTitle.c_str(),12,1,13,11,1,12);
  wheelMeanHistos[wh]->setAxisTitle("Sector",1);
  wheelMeanHistos[wh]->setBinLabel(1,"1",1);
  wheelMeanHistos[wh]->setBinLabel(2,"2",1);
  wheelMeanHistos[wh]->setBinLabel(3,"3",1);
  wheelMeanHistos[wh]->setBinLabel(4,"4",1);
  wheelMeanHistos[wh]->setBinLabel(5,"5",1);
  wheelMeanHistos[wh]->setBinLabel(6,"6",1);
  wheelMeanHistos[wh]->setBinLabel(7,"7",1);
  wheelMeanHistos[wh]->setBinLabel(8,"8",1);
  wheelMeanHistos[wh]->setBinLabel(9,"9",1);
  wheelMeanHistos[wh]->setBinLabel(10,"10",1);
  wheelMeanHistos[wh]->setBinLabel(11,"11",1);
  wheelMeanHistos[wh]->setBinLabel(12,"12",1);
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
  
  if(wheelMeanHistos.find(3) == wheelMeanHistos.end()){
    string histoName =  "MeanSummaryRes";
    histoTitle = "# of SL with |mean of res.| > " + meanRange.str() + "#mum";
    wheelMeanHistos[3] = dbe->book2D(histoName.c_str(),histoTitle.c_str(),12,1,13,5,-2,3);
    wheelMeanHistos[3]->setAxisTitle("Sector",1);
    wheelMeanHistos[3]->setBinLabel(1,"1",1);
    wheelMeanHistos[3]->setBinLabel(2,"2",1);
    wheelMeanHistos[3]->setBinLabel(3,"3",1);
    wheelMeanHistos[3]->setBinLabel(4,"4",1);
    wheelMeanHistos[3]->setBinLabel(5,"5",1);
    wheelMeanHistos[3]->setBinLabel(6,"6",1);
    wheelMeanHistos[3]->setBinLabel(7,"7",1);
    wheelMeanHistos[3]->setBinLabel(8,"8",1);
    wheelMeanHistos[3]->setBinLabel(9,"9",1);
    wheelMeanHistos[3]->setBinLabel(10,"10",1);
    wheelMeanHistos[3]->setBinLabel(11,"11",1);
    wheelMeanHistos[3]->setBinLabel(12,"12",1);
    wheelMeanHistos[3]->setAxisTitle("Wheel",2);
  }


  dbe->setCurrentFolder("DT/02-Segments/01-SigmaRes");
  histoName =  "SigmaSummaryRes_W" + wheel.str();
  stringstream sigmaRange; sigmaRange << (permittedSigmaRange*10000);
  histoTitle = "# of SL with #sigma res. > " + sigmaRange.str() + "#mum (Wheel " + wheel.str() + ")";
  wheelSigmaHistos[wh] = dbe->book2D(histoName.c_str(),histoTitle.c_str(),12,1,13,11,1,12);
  wheelSigmaHistos[wh]->setAxisTitle("Sector",1);
  wheelSigmaHistos[wh]->setBinLabel(1,"1",1);
  wheelSigmaHistos[wh]->setBinLabel(2,"2",1);
  wheelSigmaHistos[wh]->setBinLabel(3,"3",1);
  wheelSigmaHistos[wh]->setBinLabel(4,"4",1);
  wheelSigmaHistos[wh]->setBinLabel(5,"5",1);
  wheelSigmaHistos[wh]->setBinLabel(6,"6",1);
  wheelSigmaHistos[wh]->setBinLabel(7,"7",1);
  wheelSigmaHistos[wh]->setBinLabel(8,"8",1);
  wheelSigmaHistos[wh]->setBinLabel(9,"9",1);
  wheelSigmaHistos[wh]->setBinLabel(10,"10",1);
  wheelSigmaHistos[wh]->setBinLabel(11,"11",1);
  wheelSigmaHistos[wh]->setBinLabel(12,"12",1);
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

  if(wheelSigmaHistos.find(3) == wheelSigmaHistos.end()){
    string histoName =  "SigmaSummaryRes";
    histoTitle = "# of SL with #sigma res. > " + sigmaRange.str() + "#mum";
    wheelSigmaHistos[3] = dbe->book2D(histoName.c_str(),histoTitle.c_str(),14,1,15,5,-2,3);
    wheelSigmaHistos[3]->setAxisTitle("Sector",1);
    wheelSigmaHistos[3]->setBinLabel(1,"1",1);
    wheelSigmaHistos[3]->setBinLabel(2,"2",1);
    wheelSigmaHistos[3]->setBinLabel(3,"3",1);
    wheelSigmaHistos[3]->setBinLabel(4,"4",1);
    wheelSigmaHistos[3]->setBinLabel(5,"5",1);
    wheelSigmaHistos[3]->setBinLabel(6,"6",1);
    wheelSigmaHistos[3]->setBinLabel(7,"7",1);
    wheelSigmaHistos[3]->setBinLabel(8,"8",1);
    wheelSigmaHistos[3]->setBinLabel(9,"9",1);
    wheelSigmaHistos[3]->setBinLabel(10,"10",1);
    wheelSigmaHistos[3]->setBinLabel(11,"11",1);
    wheelSigmaHistos[3]->setBinLabel(12,"12",1);
    wheelSigmaHistos[3]->setBinLabel(13,"13",1);
    wheelSigmaHistos[3]->setBinLabel(14,"14",1);
    wheelSigmaHistos[3]->setAxisTitle("Wheel",2);
  }  
}


void DTResolutionAnalysisTest::bookHistos(int wh, int sect) {

  stringstream wheel; wheel << wh;		
  stringstream sector; sector << sect;	


  string MeanHistoName =  "MeanTest_W" + wheel.str() + "_Sec" + sector.str(); 
  string SigmaHistoName =  "SigmaTest_W" + wheel.str() + "_Sec" + sector.str(); 
 
  string folder = "DT/02-Segments/Wheel" + wheel.str() + "/Sector" + sector.str();
  dbe->setCurrentFolder(folder);

  if(sect!=4 && sect!=10)
    MeanHistos[make_pair(wh,sect)] = dbe->book1D(MeanHistoName.c_str(),"Mean (from gaussian fit) of the residuals distribution",11,1,12);
  else
    MeanHistos[make_pair(wh,sect)] = dbe->book1D(MeanHistoName.c_str(),"Mean (from gaussian fit) of the residuals distribution",13,1,14);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(1,"MB1_SL1",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(2,"MB1_SL2",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(3,"MB1_SL3",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(4,"MB2_SL1",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(5,"MB2_SL2",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(6,"MB2_SL3",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(7,"MB3_SL1",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(8,"MB3_SL2",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(9,"MB3_SL3",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(10,"MB4_SL1",1);
  (MeanHistos[make_pair(wh,sect)])->setBinLabel(11,"MB4_SL3",1);
  if(sect==4){
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4S4_SL1",1);
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4S4_SL3",1);
  }
  if(sect==10){
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4S10_SL1",1);
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4S10_SL3",1);
  }

  if(sect!=4 && sect!=10)
    SigmaHistos[make_pair(wh,sect)] = dbe->book1D(SigmaHistoName.c_str(),"Sigma (from gaussian fit) of the residuals distribution",11,1,12);
  else
    SigmaHistos[make_pair(wh,sect)] = dbe->book1D(SigmaHistoName.c_str(),"Sigma (from gaussian fit) of the residuals distribution",13,1,14);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(1,"MB1_SL1",1);  
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(2,"MB1_SL2",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(3,"MB1_SL3",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(4,"MB2_SL1",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(5,"MB2_SL2",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(6,"MB2_SL3",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(7,"MB3_SL1",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(8,"MB3_SL2",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(9,"MB3_SL3",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(10,"MB4_SL1",1);
  (SigmaHistos[make_pair(wh,sect)])->setBinLabel(11,"MB4_SL3",1);
  if(sect==4){
    (SigmaHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4S13_SL1",1);
    (SigmaHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4S13_SL3",1);
  }
  if(sect==10){
    (SigmaHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4S14_SL1",1);
    (SigmaHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4S14_SL3",1);
  }


}


string DTResolutionAnalysisTest::getMEName(const DTSuperLayerId & slID) {
  
  stringstream wheel; wheel << slID.wheel();	
  stringstream station; station << slID.station();	
  stringstream sector; sector << slID.sector();	
  stringstream superLayer; superLayer << slID.superlayer();
  
  string folderName = 
    folderRoot + "DT/02-Segments/Wheel" +  wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/";
  
  string histoname = folderName + "hResDist" 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str(); 
  
  return histoname;
  
}



int DTResolutionAnalysisTest::stationFromBin(int bin) const {
  return (int) (bin /3.1)+1;
}
 

int DTResolutionAnalysisTest::slFromBin(int bin) const {
  int ret = bin%3;
  if(ret == 0 || bin == 11) ret = 3;
  
  return ret;
}
