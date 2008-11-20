

/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/20 09:15:19 $
 *  $Revision: 1.3 $
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
  permittedMeanRange = ps.getUntrackedParameter<double>("permittedMeanRange",0.0005); 
  permittedSigmaRange = ps.getUntrackedParameter<double>("permittedSigmaRange",0.001); 
  
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

}


void DTResolutionAnalysisTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "[DTResolutionAnalysisTest]: "<<nevents<<" events";

}



void DTResolutionAnalysisTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  

  edm::LogVerbatim ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") <<"[DTResolutionAnalysisTest]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs 
  nLumiSegs = lumiSeg.id().luminosityBlock();
  
  // prescale factor
  if ( nLumiSegs%prescaleFactor != 0 ) return;

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
	    histo_root->Fit(gfit);
	  } catch (...) {
	    edm::LogError ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask")<< "[DTResolutionAnalysisTask]: Exception when fitting..."
									    << "SuperLayer : " << slID;
	    continue;
	  }
	  if(gfit){
	    mean = gfit->GetParameter(1); 
	    sigma = gfit->GetParameter(0);
	  }
	}
	else{
	  edm::LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask")
	    << "[DTResolutionAnalysisTask] Fit of " << slID
	    << " not performed because # entries < 20 ";
	}
	
	int BinNumber = entry+slID.superLayer();
	if(BinNumber == 12) BinNumber=11;
	if((slID.sector()==13 || slID.sector()==14)  && slID.superLayer()==1) BinNumber=12;
	if((slID.sector()==13 || slID.sector()==14) && slID.superLayer()==2) BinNumber=13;

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
	  edm::LogError("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "Bad mean channel: wh: " << wheel
									  << " st: " << stationFromBin(bin)
									  << " sect: " <<sector
									  << " sl: " << slFromBin(bin)
									  << " mean (cm): " << mean;
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
	double sigma = MeanHistos.find(make_pair(wheel,sector))->second->getBinContent(bin);
	if(sigma>permittedSigmaRange){
	  edm::LogError("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "Bad sigma: wh: " << wheel
									  << " st: " << stationFromBin(bin)
									  << " sect: " <<sector
									  << " sl: " << slFromBin(bin)
									  << " sigma (cm): " << sigma;
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
  
  dbe->setCurrentFolder("DT/02-Segments");

  stringstream wheel; wheel <<wh;
  
  string histoName =  "MeanSummaryRes_testFailed_W" + wheel.str();
  wheelMeanHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),12,1,13,11,1,12);
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
  
  histoName =  "SigmaSummaryRes_testFailed_W" + wheel.str();
  wheelSigmaHistos[wh] = dbe->book2D(histoName.c_str(),histoName.c_str(),12,1,13,11,1,12);
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
  
  if(wheelMeanHistos.find(3) == wheelMeanHistos.end()){
    string histoName =  "MeanSummaryRes_testFailed";
    wheelMeanHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),12,1,13,5,-2,3);
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
    wheelMeanHistos[3]->setBinLabel(1,"Wheel-2",2);
    wheelMeanHistos[3]->setBinLabel(2,"Wheel-1",2);
    wheelMeanHistos[3]->setBinLabel(3,"Wheel0",2);
    wheelMeanHistos[3]->setBinLabel(4,"Wheel+1",2);
    wheelMeanHistos[3]->setBinLabel(5,"Wheel+2",2);
  }

  if(wheelSigmaHistos.find(3) == wheelSigmaHistos.end()){
    string histoName =  "SigmaSummaryRes_testFailed";
    wheelSigmaHistos[3] = dbe->book2D(histoName.c_str(),histoName.c_str(),14,1,15,5,-2,3);
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
}


void DTResolutionAnalysisTest::bookHistos(int wh, int sect) {

  stringstream wheel; wheel << wh;		
  stringstream sector; sector << sect;	


  string MeanHistoName =  "MeanTest_STEP3_W" + wheel.str() + "_Sec" + sector.str(); 
  string SigmaHistoName =  "SigmaTest_STEP3_W" + wheel.str() + "_Sec" + sector.str(); 
 
  string folder = "DT/02-Segments/Wheel" + wheel.str() + "/Sector" + sector.str();
  dbe->setCurrentFolder(folder);

  if(sect!=4 && sect!=10)
    MeanHistos[make_pair(wh,sect)] = dbe->book1D(MeanHistoName.c_str(),MeanHistoName.c_str(),11,1,12);
  else
    MeanHistos[make_pair(wh,sect)] = dbe->book1D(MeanHistoName.c_str(),MeanHistoName.c_str(),13,1,14);
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
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4L_SL1",1);
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4L_SL3",1);
  }
  if(sect==10){
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4R_SL1",1);
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4R_SL3",1);
  }

  if(sect!=4 && sect!=10)
    SigmaHistos[make_pair(wh,sect)] = dbe->book1D(SigmaHistoName.c_str(),SigmaHistoName.c_str(),11,1,12);
  else
    SigmaHistos[make_pair(wh,sect)] = dbe->book1D(SigmaHistoName.c_str(),SigmaHistoName.c_str(),13,1,14);
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
    (SigmaHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4L_SL1",1);
    (SigmaHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4L_SL3",1);
  }
  if(sect==10){
    (SigmaHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4R_SL1",1);
    (SigmaHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4R_SL3",1);
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
