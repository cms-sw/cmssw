/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/03/02 19:47:32 $
 *  $Revision: 1.4 $
 *  \author A. Vilela Pereira
 */

#include "DTTTrigFillWithAverage.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

using namespace std;
using namespace edm;

namespace dtCalibration {

DTTTrigFillWithAverage::DTTTrigFillWithAverage(const ParameterSet& pset):foundAverage_(false) {
  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");
}

DTTTrigFillWithAverage::~DTTTrigFillWithAverage() {}

void DTTTrigFillWithAverage::setES(const EventSetup& setup) {
  // Get tTrig record from DB
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
  tTrigMap_ = &*tTrig;

  // Get geometry from Event Setup
  setup.get<MuonGeometryRecord>().get(muonGeom_);
}

DTTTrigData DTTTrigFillWithAverage::correction(const DTSuperLayerId& slId) {
  float tTrigMean,tTrigSigma, kFactor;
  int status = tTrigMap_->get(slId,tTrigMean,tTrigSigma,kFactor,DTTimeUnits::ns);
  if(!status){
    return DTTTrigData(tTrigMean,tTrigSigma,kFactor);
  } else {
    if(!foundAverage_) getAverage();
    float corrMean = initialTTrig_.aveMean;
    float corrSigma = initialTTrig_.aveSigma;
    float corrKFactor = initialTTrig_.aveKFactor; 
    return DTTTrigData(corrMean,corrSigma,corrKFactor); //FIXME: kFactor is not anymore a unique one
  } 
}

void DTTTrigFillWithAverage::getAverage() {
  //Get the superlayers list
  vector<DTSuperLayer*> dtSupLylist = muonGeom_->superLayers();

  float aveMean = 0.;
  float ave2Mean = 0.;
  float aveSigma = 0.;
  float ave2Sigma = 0.;
  float aveKFactor = 0.;
  int nIter = 0;
  
  for(vector<DTSuperLayer*>::const_iterator sl = muonGeom_->superLayers().begin();
                                            sl != muonGeom_->superLayers().end(); ++sl) {
    float tTrigMean,tTrigSigma,kFactor;
    int status = tTrigMap_->get((*sl)->id(),tTrigMean,tTrigSigma,kFactor,DTTimeUnits::ns);
    if(!status){
      ++nIter;
      aveMean += tTrigMean;
      ave2Mean += tTrigMean*tTrigMean;
      aveSigma += tTrigSigma;
      ave2Sigma += tTrigSigma*tTrigSigma;
      aveKFactor += kFactor;
    }
  }

  // Compute average
  aveMean /= nIter;
  float rmsMean = ave2Mean/(nIter - 1) - aveMean*aveMean;
  rmsMean = sqrt(rmsMean);
  aveSigma /= nIter;
  float rmsSigma = ave2Sigma/(nIter - 1) - aveSigma*aveSigma;
  rmsSigma = sqrt(rmsSigma);
  aveKFactor /= nIter;  

  initialTTrig_.aveMean = aveMean;
  initialTTrig_.rmsMean = rmsMean;
  initialTTrig_.aveSigma = aveSigma;
  initialTTrig_.rmsSigma = rmsSigma;
  initialTTrig_.aveKFactor = aveKFactor;

  LogVerbatim("Calibration") << "[DTTTrigFillWithAverage] Found from " << nIter << " SL's\n"
                             << "                               average tTrig mean: " << aveMean << "\n"
                             << "                               tTrig mean RMS: " << rmsMean << "\n"
                             << "                               average tTrig sigma: " << aveSigma << "\n"
                             << "                               tTrig sigma RMS: " << rmsSigma << "\n" 
                             << "                               kFactor mean: " << aveKFactor;
  foundAverage_ = true;
}

} // namespace
