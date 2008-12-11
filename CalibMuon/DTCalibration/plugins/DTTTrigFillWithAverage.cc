/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/03 09:13:59 $
 *  $Revision: 1.5 $
 *  \author A. Vilela Pereira
 */

#include "DTTTrigFillWithAverage.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

using namespace std;
using namespace edm;

DTTTrigFillWithAverage::DTTTrigFillWithAverage(const ParameterSet& pset):foundAverage_(false) {}

DTTTrigFillWithAverage::~DTTTrigFillWithAverage() {}

void DTTTrigFillWithAverage::setES(const EventSetup& setup) {
  // Get tTrig record from DB
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(tTrig);
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
    double corrMean = initialTTrig_.aveMean;
    double corrSigma = initialTTrig_.aveSigma;
    return DTTTrigData(corrMean,corrSigma,kFactor); //FIXME: kFactor is not anymore a unique one
  } 
}

void DTTTrigFillWithAverage::getAverage() {
  //Get the superlayers list
  vector<DTSuperLayer*> dtSupLylist = muonGeom_->superLayers();

  double aveMean = 0.;
  double ave2Mean = 0.;
  double aveSigma = 0.;
  double ave2Sigma = 0.;
  double nIter = 0.;
  
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
    }
  }

  // Compute average
  aveMean /= nIter;
  double rmsMean = ave2Mean/(nIter - 1) - aveMean*aveMean;
  rmsMean = sqrt(rmsMean);
  aveSigma /= nIter;
  double rmsSigma = ave2Sigma/(nIter - 1) - aveSigma*aveSigma;
  rmsSigma = sqrt(rmsSigma);

  initialTTrig_.aveMean = aveMean;
  initialTTrig_.rmsMean = rmsMean;
  initialTTrig_.aveSigma = aveSigma;
  initialTTrig_.rmsSigma = rmsSigma;

  foundAverage_ = true;
}
