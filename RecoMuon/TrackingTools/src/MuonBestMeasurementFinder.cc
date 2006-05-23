
/** \class MuonBestMeasurementFinder
 *  Algorithmic class to get best measurement from a list of TM
 *  the chi2 cut for the MeasurementEstimator is huge since should not be used.
 *  The aim of this class is to return the "best" measurement according to the
 *  chi2, but without any cut. The decision whether to use or not the
 *  measurement is taken in the caller class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

MuonBestMeasurementFinder::MuonBestMeasurementFinder(Propagator* prop):thePropagator(prop){
  theEstimator = new Chi2MeasurementEstimator(100000.);
}

MuonBestMeasurementFinder::~MuonBestMeasurementFinder(){
  delete theEstimator;
}

TrajectoryMeasurement* MuonBestMeasurementFinder::bestMeasurement(TMContainer& measC){
  
}
