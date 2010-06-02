/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/10/19 13:12:00 $
 *  $Revision: 1.2 $
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

#include <string>
#include <sstream>

using namespace std;
using namespace edm;

DTTTrigResidualCorrection::DTTTrigResidualCorrection(const ParameterSet& pset) {
  string residualsRootFile = pset.getParameter<string>("residualsRootFile");
  rootFile_ = new TFile(residualsRootFile.c_str(),"READ");
  useFit_ = pset.getParameter<bool>("useFitToResiduals");
  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");
  //useConstantvDrift_ = pset.getParameter<bool>("useConstantDriftVelocity");
}

DTTTrigResidualCorrection::~DTTTrigResidualCorrection() {
  delete rootFile_;
}

void DTTTrigResidualCorrection::setES(const EventSetup& setup) {
  // Get tTrig record from DB
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
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
    TF1 *fit = new TF1("Gaussian","gaus",-0.3,0.3); 
    fit->SetParameters(residualHisto.GetMaximum(),residualHisto.GetMean(),residualHisto.GetRMS());
    fit->SetParNames("norm","mean","width");
    residualHisto.Fit(fit,"Q0");
    fitMean = fit->GetParameter(1);
    LogTrace("Calibration") << "[DTTTrigResidualCorrection]: Fit normalization = " << fit->GetParameter(0) << "\n"
                            << "                             Mean, Fit Mean    = " << residualHisto.GetMean()
                                                                                   << ", " << fit->GetParameter(1) << "\n"
                            << "                             RMS, Fit RMS      = " << residualHisto.GetRMS() << ", " << fit->GetParameter(2);
  }

  double resTime = 0.;
  if(vDrift != 0.) resTime = ((useFit_)?fitMean:(residualHisto.GetMean()))/vDrift;

  LogTrace("Calibration") << "[DTTTrigResidualCorrection]: vDrift from DB, correction to tTrig = " << vDrift << ", " << resTime;

  double corrMean = tTrigMean;
  double corrSigma = tTrigSigma;
  double corrKFact = 0.;
  if(tTrigSigma != 0.) corrKFact = (kFactor*tTrigSigma + resTime)/tTrigSigma;

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
    "/DQMData/DT/DTCalibValidation/Wheel" + wheel.str() + 
    "/Station" + station.str() +
    "/Sector" + sector.str() +
    "/hResDist_STEP" + Step.str() +
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str() +
    "_SL" + superLayer.str();

  return histoName;
}


