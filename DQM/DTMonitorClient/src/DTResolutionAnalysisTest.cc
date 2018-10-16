/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
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
#include <cmath>


using namespace edm;
using namespace std;


DTResolutionAnalysisTest::DTResolutionAnalysisTest(const ParameterSet& ps){

  LogTrace ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "[DTResolutionAnalysisTest]: Constructor";

  prescaleFactor = ps.getUntrackedParameter<int>("diagnosticPrescale", 1);
  // permitted test range
  maxGoodMeanValue = ps.getUntrackedParameter<double>("maxGoodMeanValue",0.02); 
  minBadMeanValue = ps.getUntrackedParameter<double>("minBadMeanValue",0.04);  
  maxGoodSigmaValue = ps.getUntrackedParameter<double>("maxGoodSigmaValue",0.08); 
  minBadSigmaValue = ps.getUntrackedParameter<double>("minBadSigmaValue",0.16);
  // top folder for the histograms in DQMStore
  topHistoFolder = ps.getUntrackedParameter<string>("topHistoFolder","DT/02-Segments");

  doCalibAnalysis = ps.getUntrackedParameter<bool>("doCalibAnalysis",false);
}


DTResolutionAnalysisTest::~DTResolutionAnalysisTest(){

  LogTrace ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") << "DTResolutionAnalysisTest: analyzed " << nevents << " events";

}

  void DTResolutionAnalysisTest::beginRun(const Run& run, const EventSetup& context){

  LogTrace ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") <<"[DTResolutionAnalysisTest]: BeginRun"; 

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void DTResolutionAnalysisTest::bookHistos(DQMStore::IBooker & ibooker) {

  // global residual summary

  ibooker.setCurrentFolder(topHistoFolder);
  globalResSummary = ibooker.book2D("ResidualsGlbSummary", "# of SLs with good mean and good sigma of residuals",12,1,13,5,-2,3);

  // book summaries for mean and sigma
  ibooker.setCurrentFolder(topHistoFolder + "/00-MeanRes");
  meanDistr[-2] = ibooker.book1D("MeanDistr","Mean value of the residuals all (cm)",
      100,-0.1,0.1);
  meanDistr[-1] = ibooker.book1D("MeanDistr_Phi","Mean value of the residuals #phi SL (cm)",
      100,-0.1,0.1);
  meanDistr[0] = ibooker.book1D("MeanDistr_ThetaWh0","Mean values of the residuals #theta SL Wh 0 (cm)",
      100,-0.1,0.1);
  meanDistr[1] = ibooker.book1D("MeanDistr_ThetaWh1","Mean value of the residuals #theta SL Wh +/-1 (cm)",
      100,-0.1,0.1);
  meanDistr[2] = ibooker.book1D("MeanDistr_ThetaWh2","Mean value of the residuals #theta SL Wh +/-2 (cm)",
      100,-0.1,0.1);


  string histoTitle = "# of SLs with good mean of residuals";

  wheelMeanHistos[3] = ibooker.book2D("MeanResGlbSummary",histoTitle.c_str(),12,1,13,5,-2,3);
  wheelMeanHistos[3]->setAxisTitle("Sector",1);
  wheelMeanHistos[3]->setAxisTitle("Wheel",2);

  ibooker.setCurrentFolder(topHistoFolder + "/01-SigmaRes");
  sigmaDistr[-2] = ibooker.book1D("SigmaDistr","Sigma value of the residuals all (cm)",
      50,0.0,0.2);
  sigmaDistr[-1] = ibooker.book1D("SigmaDistr_Phi","Sigma value of the residuals #phi SL (cm)",
      50,0.0,0.2);
  sigmaDistr[0] = ibooker.book1D("SigmaDistr_ThetaWh0","Sigma value of the residuals #theta SL Wh 0 (cm)",
      50,0.0,0.2);
  sigmaDistr[1] = ibooker.book1D("SigmaDistr_ThetaWh1","Sigma value of the residuals #theta SL Wh +/-1 (cm)",
      50,0.0,0.2);
  sigmaDistr[2] = ibooker.book1D("SigmaDistr_ThetaWh2","Sigma value of the residuals #theta SL Wh +/-2 (cm)",
      50,0.0,0.2);

  histoTitle = "# of SLs with good sigma of residuals";
  wheelSigmaHistos[3] = ibooker.book2D("SigmaResGlbSummary",histoTitle.c_str(),12,1,13,5,-2,3);
  wheelSigmaHistos[3]->setAxisTitle("Sector",1);
  wheelSigmaHistos[3]->setAxisTitle("Wheel",2);


  // loop over all the CMS wheels, sectors & book the summary histos
  for (int wheel=-2; wheel<=2; wheel++){
    bookHistos(ibooker,wheel);
    for (int sector=1; sector<=12; sector++){
      bookHistos(ibooker,wheel, sector);
    }
  }

}

void DTResolutionAnalysisTest::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  if (!igetter.dirExists(topHistoFolder)) {
    LogTrace ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") 
      <<"[DTResolutionAnalysisTest]: Base folder " << topHistoFolder 
      << " does not exist. Skipping client operation." << endl;
    return;
  }

  bookHistos(ibooker); // histos booked only if top histo folder exist
  // as Standard/AlcaReco Harvest is performed in the same step

  LogTrace ("DTDQM|DTMonitorClient|DTResolutionAnalysisTest") 
    << "[DTResolutionAnalysisTest]: End of Run transition, performing the DQM client operation" << endl;

  // reset the ME with fixed scale
  resetMEs();

  for (vector<const DTChamber*>::const_iterator ch_it = muonGeom->chambers().begin();
      ch_it != muonGeom->chambers().end(); ++ch_it) {  // loop over the chambers

    DTChamberId chID = (*ch_it)->id();

    // Fill the test histos
    for(vector<const DTSuperLayer*>::const_iterator sl_it = (*ch_it)->superLayers().begin();
        sl_it != (*ch_it)->superLayers().end(); ++sl_it) {    // loop over SLs


      DTSuperLayerId slID = (*sl_it)->id();
      MonitorElement * res_histo = igetter.get(getMEName(slID));

      if(res_histo) { // Gaussian Fit
        float statMean = res_histo->getMean(1);
        float statSigma = res_histo->getRMS(1);
        double mean = -1;
        double sigma = -1;
        TH1F * histo_root = res_histo->getTH1F();

        // fill the summaries
        int entry= (chID.station() - 1) * 3;
        int binSect = slID.sector();
        if(slID.sector() == 13) binSect = 4;
        else if(slID.sector() == 14) binSect = 10;
        int binSL = entry+slID.superLayer();
        if(chID.station() == 4 && slID.superLayer() == 3) binSL--;
        if((slID.sector()==13 || slID.sector()==14)  && slID.superLayer()==1) binSL=12;
        if((slID.sector()==13 || slID.sector()==14) && slID.superLayer()==3) binSL=13;

        if(histo_root->GetEntries()>20) {
          TF1 *gfit = new TF1("Gaussian","gaus",(statMean-(2*statSigma)),(statMean+(2*statSigma)));
          try {
            histo_root->Fit(gfit, "Q0 SERIAL", "", -0.1, 0.1);
          } catch (cms::Exception& iException) {
            LogWarning ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask")
              << "[DTResolutionAnalysisTask]: Exception when fitting SL : " << slID;
            // FIXME: the SL is set as OK in the summary
            double weight = 1/11.;
            if((binSect == 4 || binSect == 10) && slID.station() == 4)  weight = 1/22.;
            globalResSummary->Fill(binSect, slID.wheel(), weight);
            continue;
          }

          if(gfit){
            // get the mean and the sigma of the distribution
            mean = gfit->GetParameter(1); 
            sigma = gfit->GetParameter(2);

            // fill the distributions
            meanDistr[-2]->Fill(mean);
            sigmaDistr[-2]->Fill(sigma);
            if(slID.superlayer() == 2) {
              meanDistr[abs(slID.wheel())]->Fill(mean);
              sigmaDistr[abs(slID.wheel())]->Fill(sigma);
            } else {
              meanDistr[-1]->Fill(mean);
              sigmaDistr[-1]->Fill(sigma);
            }

            // sector summaries
            MeanHistos[make_pair(slID.wheel(),binSect)]->setBinContent(binSL, mean);	
            SigmaHistos[make_pair(slID.wheel(),binSect)]->setBinContent(binSL, sigma);

            if((slID.sector() == 13 || slID.sector() == 14) && binSL == 12) binSL=10;
            if((slID.sector() == 13 || slID.sector() == 14) && binSL == 13) binSL=11;


            if((slID.sector() == 13 || slID.sector() == 14) ) {

              double MeanVal = wheelMeanHistos[slID.wheel()]->getBinContent(binSect,binSL);
              double MeanBinVal = (MeanVal > 0. && MeanVal < meanInRange(mean)) ? MeanVal : meanInRange(mean);
              wheelMeanHistos[slID.wheel()]->setBinContent(binSect,binSL,MeanBinVal);

              double SigmaVal =  wheelSigmaHistos[slID.wheel()]->getBinContent(binSect,binSL);
              double SigmaBinVal = (SigmaVal > 0. && SigmaVal < sigmaInRange(sigma)) ? SigmaVal : sigmaInRange(sigma);
              wheelSigmaHistos[slID.wheel()]->setBinContent(binSect,binSL,SigmaBinVal);

            } else {
              wheelMeanHistos[slID.wheel()]->setBinContent(binSect,binSL,meanInRange(mean));
              wheelSigmaHistos[slID.wheel()]->setBinContent(binSect,binSL,sigmaInRange(sigma));
            }

            // set the weight
            double weight = 1/11.;
            if((binSect == 4 || binSect == 10) && slID.station() == 4)  weight = 1/22.;

            // test the values of mean and sigma
            if( (meanInRange(mean) > 0.85) && (sigmaInRange(sigma) > 0.85) ) { // sigma and mean ok
              globalResSummary->Fill(binSect, slID.wheel(), weight);
              wheelMeanHistos[3]->Fill(binSect,slID.wheel(),weight);
              wheelSigmaHistos[3]->Fill(binSect,slID.wheel(),weight);
            } else {
              if( (meanInRange(mean) < 0.85) && (sigmaInRange(sigma) > 0.85) ) { // only sigma ok
                wheelSigmaHistos[3]->Fill(binSect,slID.wheel(),weight);
              }
              if((meanInRange(mean) > 0.85) && (sigmaInRange(sigma) < 0.85)  ) { // only mean ok
                wheelMeanHistos[3]->Fill(binSect,slID.wheel(),weight);
              }
            }
          }
          delete gfit;
        }
        else{
          LogVerbatim ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask")
            << "[DTResolutionAnalysisTask] Fit of " << slID
            << " not performed because # entries < 20 ";
          // FIXME: the SL is set as OK in the summary
          double weight = 1/11.;
          if((binSect == 4 || binSect == 10) && slID.station() == 4)  weight = 1/22.;
          globalResSummary->Fill(binSect, slID.wheel(), weight);
          wheelMeanHistos[3]->Fill(binSect,slID.wheel(),weight);
          wheelSigmaHistos[3]->Fill(binSect,slID.wheel(),weight);
          wheelMeanHistos[slID.wheel()]->setBinContent(binSect,binSL,1.);
          wheelSigmaHistos[slID.wheel()]->setBinContent(binSect,binSL,1.);
        }
      } else {
        LogWarning ("DTDQM|DTMonitorModule|DTResolutionAnalysisTask")
          << "[DTResolutionAnalysisTask] Histo: " << getMEName(slID) << " not found" << endl;
      }
    } // loop on SLs
  } // Loop on Stations

}

void DTResolutionAnalysisTest::bookHistos(DQMStore::IBooker & ibooker,int wh) { 

  stringstream wheel; wheel <<wh;

  ibooker.setCurrentFolder(topHistoFolder + "/00-MeanRes");
  string histoName =  "MeanSummaryRes_W" + wheel.str();
  string histoTitle = "# of SLs with wrong mean of residuals (Wheel " + wheel.str() + ")";

  wheelMeanHistos[wh] = ibooker.book2D(histoName.c_str(),histoTitle.c_str(),12,1,13,11,1,12);
  wheelMeanHistos[wh]->setAxisTitle("Sector",1);
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

  ibooker.setCurrentFolder(topHistoFolder + "/01-SigmaRes");
  histoName =  "SigmaSummaryRes_W" + wheel.str();
  histoTitle = "# of SLs with wrong sigma of residuals (Wheel " + wheel.str() + ")";

  wheelSigmaHistos[wh] = ibooker.book2D(histoName.c_str(),histoTitle.c_str(),12,1,13,11,1,12);
  wheelSigmaHistos[wh]->setAxisTitle("Sector",1);
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

void DTResolutionAnalysisTest::bookHistos(DQMStore::IBooker & ibooker,int wh, int sect) {

  stringstream wheel; wheel << wh;		
  stringstream sector; sector << sect;	


  string MeanHistoName =  "MeanTest_W" + wheel.str() + "_Sec" + sector.str(); 
  string SigmaHistoName =  "SigmaTest_W" + wheel.str() + "_Sec" + sector.str(); 

  string folder = topHistoFolder + "/Wheel" + wheel.str() + "/Sector" + sector.str();
  ibooker.setCurrentFolder(folder);

  if(sect!=4 && sect!=10) {
    MeanHistos[make_pair(wh,sect)] =

      ibooker.book1D(MeanHistoName.c_str(),"Mean (from gaussian fit) of the residuals distribution",11,1,12);
  } else {
    MeanHistos[make_pair(wh,sect)] =
      ibooker.book1D(MeanHistoName.c_str(),"Mean (from gaussian fit) of the residuals distribution",13,1,14);
  }
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
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4S13_SL1",1);
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4S13_SL3",1);
  }
  if(sect==10){
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(12,"MB4S14_SL1",1);
    (MeanHistos[make_pair(wh,sect)])->setBinLabel(13,"MB4S14_SL3",1);
  }

  if(sect!=4 && sect!=10) {
    SigmaHistos[make_pair(wh,sect)] =
      ibooker.book1D(SigmaHistoName.c_str(),"Sigma (from gaussian fit) of the residuals distribution",11,1,12);
  } else {
    SigmaHistos[make_pair(wh,sect)] =
      ibooker.book1D(SigmaHistoName.c_str(),"Sigma (from gaussian fit) of the residuals distribution",13,1,14);
  }
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
    topHistoFolder + "/Wheel" +  wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/";

  if(doCalibAnalysis) folderName =
    "DT/DTCalibValidation/Wheel" +  wheel.str() +
      "/Station" + station.str() + "/Sector" + sector.str() + "/";

  string histoname = folderName + "hResDist" 
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() 
    + "_SL" + superLayer.str(); 

  if(doCalibAnalysis) histoname = folderName + "hResDist_STEP3" 
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

double DTResolutionAnalysisTest::meanInRange(double mean) const {
  double value(0.);
  if( fabs(mean) <= maxGoodMeanValue ) {value = 1.;}
  else if( fabs(mean) > maxGoodMeanValue && fabs(mean) < minBadMeanValue ) {value = 0.9;}
  else if( fabs(mean) >= minBadMeanValue ) {value = 0.1;}
  return value;
}

double DTResolutionAnalysisTest::sigmaInRange(double sigma) const {
  double value(0.);
  if( sigma <= maxGoodSigmaValue ) {value = 1.;}
  else if( sigma > maxGoodSigmaValue && sigma < minBadSigmaValue ) {value = 0.9;}
  else if( sigma >= minBadSigmaValue ) {value = 0.1;}
  return value;
}

void DTResolutionAnalysisTest::resetMEs() {
  globalResSummary->Reset();
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

  for(int indx = -2; indx != 3; ++indx) {
    meanDistr[indx]->Reset();
    sigmaDistr[indx]->Reset();
  }
}

