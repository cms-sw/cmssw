
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/11/18 20:59:09 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "DTVDriftSegment.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"

#include "CalibMuon/DTCalibration/interface/DTResidualFitter.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include <string>
#include <vector>

#include "TH1F.h"
#include "TFile.h"

using namespace std;
using namespace edm;

DTVDriftSegment::DTVDriftSegment(const ParameterSet& pset) {
  string rootFileName = pset.getParameter<string>("rootFileName");
  rootFile_ = new TFile(rootFileName.c_str(), "READ");

  fitter_ = new DTResidualFitter();
  //bool debug = pset.getUntrackedParameter<bool>("debug", false);
  //if(debug) fitter_->setVerbosity(1);
}

DTVDriftSegment::~DTVDriftSegment() {
  rootFile_->Close();
  delete fitter_;
}

void DTVDriftSegment::setES(const edm::EventSetup& setup) {
  // Get the map of ttrig from the Setup
  ESHandle<DTMtime> mTime;
  setup.get<DTMtimeRcd>().get(mTime);
  mTimeMap_ = &*mTime;
}

DTVDriftData DTVDriftSegment::compute(DTSuperLayerId const& slId) {

  // Get original value from DB; vdrift is cm/ns , resolution is cm
  float vDrift = 0., resolution = 0.;
  int status = mTimeMap_->get(slId,vDrift,resolution,DTVelocityUnits::cm_per_ns);

   if(status != 0) throw cms::Exception("DTCalibration") << "Could not find vDrift entry in DB for"
                                                         << slId << endl;
  // For RZ superlayers use original value
  if(slId.superLayer() == 2){
     return DTVDriftData(vDrift,resolution);
  } else{
     TH1F* vDriftCorrHisto = getHisto(slId);
     int nSigmas = 1;
     DTResidualFitResult fitResult = fitter_->fitResiduals(*vDriftCorrHisto,nSigmas);
     LogTrace("Calibration") << "[DTVDriftSegment]: \n"
                             << " Fit Mean  = " << fitResult.fitMean << " +/- " << fitResult.fitMeanError << "\n"
                             << " Fit Sigma = " << fitResult.fitSigma << " +/- " << fitResult.fitSigmaError;

     float vDriftCorr = fitResult.fitMean;
     float vDriftNew = vDrift*(1. - vDriftCorr); 
     float resolutionNew = resolution;
     return DTVDriftData(vDriftNew,resolutionNew);
  }
}

TH1F* DTVDriftSegment::getHisto(const DTSuperLayerId& slId) {
  string histoName = getHistoName(slId);
  TH1F* histo = static_cast<TH1F*>(rootFile_->Get(histoName.c_str()));
  if(!histo) throw cms::Exception("DTCalibration") << "v-drift correction histogram not found:"
                                                   << histoName << endl; 
  return histo;
}

string DTVDriftSegment::getHistoName(const DTSuperLayerId& slId) {
  DTChamberId chId = slId.chamberId();

  // Compose the chamber name
  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();
  stringstream sector; sector << chId.sector();

  string chHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();

  return (slId.superLayer() != 2)?("hRPhiVDriftCorr" + chHistoName):("hRZVDriftCorr" + chHistoName);
}

