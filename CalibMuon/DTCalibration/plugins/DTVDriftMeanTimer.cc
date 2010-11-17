
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/09 22:44:10 $
 *  $Revision: 1.7 $
 *  \author A. Vilela Pereira
 */

#include "DTVDriftMeanTimer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"

#include "CalibMuon/DTCalibration/interface/DTMeanTimerFitter.h"
#include "CalibMuon/DTCalibration/interface/vDriftHistos.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include <string>
#include <vector>

#include "TFile.h"
#include "TString.h"

using namespace std;
using namespace edm;

DTVDriftMeanTimer::DTVDriftMeanTimer(const ParameterSet& pset) {
  string rootFileName = pset.getParameter<string>("rootFileName");
  rootFile_ = new TFile(rootFileName.c_str(), "READ");
  fitter_ = new DTMeanTimerFitter(rootFile_);
  bool debug = pset.getUntrackedParameter<bool>("debug", false);
  if(debug) fitter_->setVerbosity(1);
}

DTVDriftMeanTimer::~DTVDriftMeanTimer() {
  rootFile_->Close();
  delete fitter_;
}

void DTVDriftMeanTimer::setES(const edm::EventSetup& setup) {}

DTVDriftData DTVDriftMeanTimer::compute(DTSuperLayerId const& slId) {

  // Evaluate v_drift and sigma from the TMax histograms
  DTWireId wireId(slId, 0, 0);
  TString N = ( ( ( ( (TString)"TMax" + (long)wireId.wheel() )
                                      + (long)wireId.station() )
		                      + (long)wireId.sector() ) + (long)wireId.superLayer() );
  vector<float> vDriftAndReso = fitter_->evaluateVDriftAndReso(N);

  // Don't write the constants for the SL if the vdrift was not computed
  if(vDriftAndReso.front() == -1)
     throw cms::Exception("DTCalibration") << "Could not compute valid vDrift value for SL " << slId << endl;

  return DTVDriftData(vDriftAndReso[0],vDriftAndReso[1]);
}
