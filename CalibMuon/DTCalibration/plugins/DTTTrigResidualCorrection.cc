/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/06/29 15:42:03 $
 *  $Revision: 1.5 $
 *  \author A. Vilela Pereira
 */

#include "DTTTrigResidualCorrection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"

#include "TFile.h"
#include "TH1F.h"
#include "TF1.h"
#include "TCanvas.h"

#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "RooAddPdf.h"
#include "RooFitResult.h"
#include "RooGlobalFunc.h"

#include <string>
#include <sstream>

using namespace std;
using namespace edm;

DTTTrigResidualCorrection::DTTTrigResidualCorrection(const ParameterSet& pset) {
  string residualsRootFile = pset.getParameter<string>("residualsRootFile");
  rootFile_ = new TFile(residualsRootFile.c_str(),"READ");
  rootBaseDir_ = pset.getUntrackedParameter<string>("rootBaseDir","/DQMData/DT/DTCalibValidation"); 
  useFit_ = pset.getParameter<bool>("useFitToResiduals");
  //useConstantvDrift_ = pset.getParameter<bool>("useConstantDriftVelocity");
  dbLabel_  = pset.getUntrackedParameter<string>("dbLabel", "");
  useSlopesCalib_ = pset.getUntrackedParameter<bool>("useSlopesCalib",false);

  // Load external slopes
  if(useSlopesCalib_){ 
     ifstream fileSlopes; 
     fileSlopes.open( pset.getParameter<FileInPath>("slopesFileName").fullPath().c_str() );
     //fin.open("slopesDB.txt");

     int tmp_wheel = 0;
     int tmp_sector = 0;
     int tmp_station = 0;
     int tmp_SL = 0;
     double tmp_ttrig = 0.;
     double tmp_t0 = 0.;
     double tmp_kfact = 0.;
     int tmp_a = 0;
     int tmp_b = 0;
     int tmp_c = 0;
     int tmp_d = 0;
     double tmp_v_eff = 0.;

     while(!fileSlopes.eof()){

        fileSlopes >> tmp_wheel >> tmp_sector >> tmp_station >> tmp_SL  >> tmp_a >> tmp_b >>
                      tmp_ttrig >> tmp_t0 >> tmp_kfact >> tmp_c >> tmp_d >> tmp_v_eff;

        vDriftEff_[tmp_wheel+2][tmp_sector-1][tmp_station-1][tmp_SL-1] = -tmp_v_eff;

     }

     fileSlopes.close();
  } 

}

DTTTrigResidualCorrection::~DTTTrigResidualCorrection() {
  delete rootFile_;
}

void DTTTrigResidualCorrection::setES(const EventSetup& setup) {
  // Get tTrig record from DB
  ESHandle<DTTtrig> tTrig;
  //setup.get<DTTtrigRcd>().get(tTrig);
  setup.get<DTTtrigRcd>().get(dbLabel_,tTrig);
  tTrigMap_ = &*tTrig;

  // Get vDrift record
  ESHandle<DTMtime> mTimeHandle;
  setup.get<DTMtimeRcd>().get(mTimeHandle);
  mTimeMap_ = &*mTimeHandle;

}

DTTTrigData DTTTrigResidualCorrection::correction(const DTSuperLayerId& slId) {

  float tTrigMean,tTrigSigma,kFactor;
  int status = tTrigMap_->get(slId,tTrigMean,tTrigSigma,kFactor,DTTimeUnits::ns);
  if(status != 0) throw cms::Exception("[DTTTrigResidualCorrection]") << "Could not find tTrig entry in DB for"
                                                                      << slId << endl;

  float vDrift,hitResolution;
  status = mTimeMap_->get(slId,vDrift,hitResolution,DTVelocityUnits::cm_per_ns);
  if(status != 0) throw cms::Exception("[DTTTrigResidualCorrection]") << "Could not find vDrift entry in DB for"
                                                                      << slId << endl;
  TH1F residualHisto = *(getHisto(slId));
  double fitMean = -1.;

  if(useFit_){

    RooRealVar x("x","residual",-1.,1.);
    RooRealVar mean("mean","mean",residualHisto.GetMean(),-0.3,0.3);
    RooRealVar sigma1("sigma1","sigma1",0.,0.5);
    RooRealVar sigma2("sigma2","sigma2",0.,0.5);

    RooRealVar frac("frac","frac",0.,1.);

    RooGaussian myg1("myg1","Gaussian distribution",x,mean,sigma1);
    RooGaussian myg2("myg2","Gaussian distribution",x,mean,sigma2);

    RooAddPdf myg("myg","myg",RooArgList(myg1,myg2),RooArgList(frac));

    RooDataHist hdata("hdata","Binned data",RooArgList(x),&residualHisto);
    myg.fitTo(hdata,RooFit::Minos(0),RooFit::Range(-0.2,0.2));

    fitMean = mean.getVal();
    LogTrace("Calibration") << "[DTTTrigResidualCorrection]: \n"
                            << "                             Mean, Fit Mean    = " << residualHisto.GetMean()
			    << ", " << fitMean << "\n"
                            << "                             RMS, Fit RMS      = " << residualHisto.GetRMS()
                            << ", " << sigma1.getVal();

    //static int count = 0;
    /*
    if(count == 0){
      RooPlot *xframe = x.frame();
      hdata.plotOn(xframe);
      myg.plotOn(xframe);
      TCanvas c1;
      c1.cd();
      xframe->Draw();
      c1.SaveAs("prova.eps");
      count++;
    }
    */

  }

  int wheel = slId.wheel();
  int sector = slId.sector();
  int station = slId.station();
  int superLayer = slId.superLayer();

  double resTime = 0.;
  if(useSlopesCalib_){
     double vdrift_eff = vDriftEff_[wheel+2][sector-1][station-1][superLayer-1];
     if(vdrift_eff == 0) vdrift_eff = vDrift;

     if(vdrift_eff) resTime = ( (useFit_) ? fitMean : residualHisto.GetMean() )/vdrift_eff;

     LogTrace("Calibration") << "[DTTTrigResidualCorrection]: Effective vDrift, correction to tTrig = "
                             << vdrift_eff << ", " << resTime;
  } else{
     if(vDrift) resTime = ( (useFit_) ? fitMean : residualHisto.GetMean() )/vDrift;

     LogTrace("Calibration") << "[DTTTrigResidualCorrection]: vDrift from DB, correction to tTrig = "
                             << vDrift << ", " << resTime;
  }
 
  double corrMean = tTrigMean;
  double corrSigma = (tTrigSigma != 0.) ? tTrigSigma : 1.;
  double corrKFact = (tTrigSigma != 0.) ? (kFactor + resTime/tTrigSigma) : resTime;

  return DTTTrigData(corrMean,corrSigma,corrKFact);  
}

const TH1F* DTTTrigResidualCorrection::getHisto(const DTSuperLayerId& slId) {
  string histoName = getHistoName(slId);
  LogTrace("Calibration") << "[DTTTrigResidualCorrection]: Accessing histogram " << histoName.c_str();
  TH1F* histo = static_cast<TH1F*>(rootFile_->Get(histoName.c_str()));
  if(!histo) throw cms::Exception("[DTTTrigResidualCorrection]") << "residual histogram not found:"
                                                                 << histoName << endl; 
  return histo;
}

string DTTTrigResidualCorrection::getHistoName(const DTSuperLayerId& slId) {

  int step = 3;
  stringstream wheel; wheel << slId.wheel();	
  stringstream station; station << slId.station();	
  stringstream sector; sector << slId.sector();	
  stringstream superLayer; superLayer << slId.superlayer();
  stringstream Step; Step << step;

  string histoName =
    rootBaseDir_ + "/Wheel" + wheel.str() + 
    "/Station" + station.str() +
    "/Sector" + sector.str() +
    "/hResDist_STEP" + Step.str() +
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str() +
    "_SL" + superLayer.str();

  return histoName;
}


