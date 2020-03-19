///=== This is the base class for the linearised chi-squared track fit algorithms.

///=== Written by: Sioni Summers and Alexander D. Morton

#include "L1Trigger/TrackFindingTMTT/interface/L1ChiSquared.h"
#include "L1Trigger/TrackFindingTMTT/interface/Matrix.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
 
#include <algorithm>
#include <functional>
#include <set>
 
namespace TMTT {

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b){
    assert(a.size() == b.size());
    std::vector<T> result;
    result.reserve(a.size());
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result), std::minus<T>());
    return result;
}
 
L1ChiSquared::L1ChiSquared(const Settings* settings, const uint nPar) : TrackFitGeneric(settings), chiSq_ (0.0){
  // Bad stub killing settings
  numFittingIterations_ = getSettings()->numTrackFitIterations();
  killTrackFitWorstHit_ = getSettings()->killTrackFitWorstHit();
  generalResidualCut_   = getSettings()->generalResidualCut(); // The cut used to remove bad stubs (if nStubs > minLayers)
  killingResidualCut_   = getSettings()->killingResidualCut(); // The cut used to kill off tracks entirely
  
  //--- These two parameters are used to check if after the fit, there are still enough stubs on the track
  minStubLayers_ = getSettings()->minStubLayers();
  nPar_ = nPar;
}
 
void L1ChiSquared::calculateChiSq( std::vector<double> resids ){
  chiSq_ = 0.0;
  uint j=0;
  for ( uint i=0; i<stubs_.size(); i++ ){
    chiSq_+= resids[j]*resids[j]+resids[j+1]*resids[j+1];
    j=j+2;
  }
}

void L1ChiSquared::calculateDeltaChiSq( std::vector<double> delX, std::vector<double> covX ){
  for ( uint i=0; i<covX.size(); i++ ){
    chiSq_ += (-delX[i])*covX[i];
  }
}

L1fittedTrack L1ChiSquared::fit(const L1track3D& l1track3D){
  
  stubs_ = l1track3D.getStubs();

  // Get cut on number of layers including variation due to dead sectors, pt dependence etc.
  minStubLayersRed_ = Utility::numLayerCut("FIT", getSettings(), l1track3D.iPhiSec(), l1track3D.iEtaReg(), fabs(l1track3D.qOverPt()), l1track3D.eta());
  
  std::vector<double> x = seed(l1track3D);
  
  Matrix<double> d = D(x);
  Matrix<double> dtVinv = d.transpose() * Vinv();
//  Matrix<double> M = dtVinv * d;
  Matrix<double> M = dtVinv * (dtVinv.transpose()); //TODO this match tracklet code, but not literature:w

  std::vector<double> resids = residuals(x);
//  std::cout << "resids.size(): " << resids.size() << std::endl;
  std::vector<double> deltaX = M.inverse() * dtVinv * resids;
  x = x - deltaX;
  std::vector<double> covX = d.transpose() * Vinv() * resids;  

  calculateChiSq(resids);
  calculateDeltaChiSq (deltaX, covX);
  resids = residuals(x); // update resids.

  for (int i=1;i<numFittingIterations_+1;++i) {
    if (i>1) {
      /*
      // Original buggy code of 18th April 2018
      if ( killTrackFitWorstHit_  &&  (largestresid_ > killingResidualCut_ || (largestresid_ > generalResidualCut_ && Utility::countLayers( getSettings(), stubs_ ) > minStubLayersRed_)) ) {
	cout<<"RATS KILLED STUB "<<largestresid_<<" > "<<killingResidualCut_<<" or "<<generalResidualCut_<<" ; "<<Utility::countLayers( getSettings(), stubs_ )<<" > "<<minStubLayersRed_<<endl;
        stubs_.erase(stubs_.begin()+ilargestresid_);
        if (getSettings()->debug() == 6) std::cout << __FILE__ " : Killed stub " << ilargestresid_ << "." << std::endl;
      }
      */

      // IRT - Debugged code of 19th April 2018
      bool killWorstStub = false;
      if (killTrackFitWorstHit_) {
	if (largestresid_ > killingResidualCut_) {
	  killWorstStub = true;
	} else if (largestresid_ > generalResidualCut_) {
	  std::vector<const Stub*> stubsTmp = stubs_;
	  stubsTmp.erase(stubsTmp.begin()+ilargestresid_);
	  if (Utility::countLayers( getSettings(), stubsTmp ) >= minStubLayersRed_) killWorstStub = true;
	} else {
	  // Get better apparent tracking performance by always killing worst stub until only 4 layers left.
          if (Utility::countLayers( getSettings(), stubs_ ) > minStubLayersRed_) killWorstStub = true;
        }
      }

      if (killWorstStub) {
        stubs_.erase(stubs_.begin()+ilargestresid_);
        if (getSettings()->debug() == 6) std::cout << __FILE__ " : Killed stub " << ilargestresid_ << "." << std::endl;

	// Reject tracks with too many killed stubs & stop iterating.
	unsigned int nLayers = Utility::countLayers( getSettings(), stubs_ ); // Count tracker layers with stubs
	bool valid = nLayers >= minStubLayersRed_;

	if ( ! valid ){
	  const unsigned int hitPattern = 0; // FIX: Needs setting
	  return L1fittedTrack (getSettings(), l1track3D, stubs_, hitPattern, l1track3D.qOverPt(), 0., l1track3D.phi0(), l1track3D.z0(), l1track3D.tanLambda(), 999999., 4, valid);
	}
      }

      d = D(x); // Calculate derivatives
      dtVinv = d.transpose() * Vinv();
      M = dtVinv * (dtVinv.transpose()); 
      resids = residuals(x); // Calculate new residuals
      std::vector<double> deltaX = M.inverse() * dtVinv * resids;
      x = x - deltaX;
      std::vector<double> covX = d.transpose() * Vinv() * resids;  
      resids = residuals(x); // update resids.

      calculateChiSq(resids);
      calculateDeltaChiSq (deltaX, covX);
    }
  }  

  std::map<std::string, double> tp = convertParams(x); // tp = track params

  // Reject tracks with too many killed stubs
  unsigned int nLayers = Utility::countLayers( getSettings(), stubs_ ); // Count tracker layers with stubs
  bool valid4par = nLayers >= minStubLayersRed_;

  const unsigned int hitPattern = 0; // FIX: Needs setting
  if ( valid4par ){
    return L1fittedTrack(getSettings(), l1track3D, stubs_, hitPattern, tp["qOverPt"], 0, tp["phi0"], tp["z0"], tp["t"], chiSq_, nPar_, valid4par);
  }
  else{ 
    return L1fittedTrack (getSettings(), l1track3D, stubs_, hitPattern, l1track3D.qOverPt(), 0., l1track3D.phi0(), l1track3D.z0(), l1track3D.tanLambda(), 999999., 4, valid4par);
  }
}
 

}
