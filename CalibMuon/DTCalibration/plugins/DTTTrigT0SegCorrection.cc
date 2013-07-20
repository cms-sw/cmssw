/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/03/02 19:47:33 $
 *  $Revision: 1.3 $
 *  \author A. Vilela Pereira
 */

#include "DTTTrigT0SegCorrection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include "TFile.h"
#include "TH1F.h"

#include <string>
#include <sstream>

using namespace std;
using namespace edm;

namespace dtCalibration {

DTTTrigT0SegCorrection::DTTTrigT0SegCorrection(const ParameterSet& pset) {
  string t0SegRootFile = pset.getParameter<string>("t0SegRootFile");
  rootFile_ = new TFile(t0SegRootFile.c_str(),"READ");
  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");
}

DTTTrigT0SegCorrection::~DTTTrigT0SegCorrection() {
  delete rootFile_;
}

void DTTTrigT0SegCorrection::setES(const EventSetup& setup) {
  // Get tTrig record from DB
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
  tTrigMap_ = &*tTrig;
}

DTTTrigData DTTTrigT0SegCorrection::correction(const DTSuperLayerId& slId) {
  float tTrigMean,tTrigSigma,kFactor;
  int status = tTrigMap_->get(slId,tTrigMean,tTrigSigma,kFactor,DTTimeUnits::ns);
  if(status != 0) throw cms::Exception("[DTTTrigT0SegCorrection]") << "Could not find tTrig entry in DB for"
                                                                   << slId << endl;

  const TH1F* t0SegHisto = getHisto(slId);
  double corrMean = tTrigMean;
  double corrSigma = tTrigSigma;
  //FIXME: can we fit the t0seg histo? How do we remove the peak at 0?;
  double corrKFact = (kFactor*tTrigSigma + t0SegHisto->GetMean())/tTrigSigma;
  return DTTTrigData(corrMean,corrSigma,corrKFact);  
}

const TH1F* DTTTrigT0SegCorrection::getHisto(const DTSuperLayerId& slId) {
  string histoName = getHistoName(slId);
  TH1F* histo = static_cast<TH1F*>(rootFile_->Get(histoName.c_str()));
  if(!histo) throw cms::Exception("[DTTTrigT0SegCorrection]") << "t0-seg histogram not found:"
                                                              << histoName << endl; 
  return histo;
}

string DTTTrigT0SegCorrection::getHistoName(const DTSuperLayerId& slId) {
  DTChamberId chId = slId.chamberId();

  // Compose the chamber name
  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();
  stringstream sector; sector << chId.sector();

  string chHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();

  return (slId.superLayer() != 2)?("hRPhiSegT0"+chHistoName):("hRZSegT0"+chHistoName);
}

} // namespace
