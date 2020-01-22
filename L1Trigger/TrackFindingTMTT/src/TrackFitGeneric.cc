///=== This is the base class for all the track fit algorithms

///=== Written by: Alexander D. Morton and Sioni Summers

#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "L1Trigger/TrackFindingTMTT/interface/ChiSquared4ParamsApprox.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/SimpleLR.h"
#ifdef USE_HLS
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFParamsCombCallHLS.h"
#endif
#include "FWCore/Utilities/interface/Exception.h"
#include <map> 
#include <new>
 
namespace TMTT {

//=== Set configuration parameters.
 
TrackFitGeneric::TrackFitGeneric( const Settings* settings, const string &fitterName ) : settings_(settings), fitterName_(fitterName), nDupStubs_(0) {
}
 
 
//=== Fit a track candidate obtained from the Hough Transform.
//=== Specify which phi sector and eta region it is in.
 
L1fittedTrack TrackFitGeneric::fit(const L1track3D& l1track3D) {
  return L1fittedTrack (settings_, l1track3D, l1track3D.getStubs(), 0, 0, 0, 0, 0, 0, 0, 999999., 0);
}
 
TrackFitGeneric* TrackFitGeneric::create(std::string fitter, const Settings* settings) {
    if (fitter.compare("ChiSquared4ParamsApprox")==0) {
	return new ChiSquared4ParamsApprox(settings, 4);
    } else if (fitter.compare("KF4ParamsComb")==0) {
	return new KFParamsComb(settings, 4, fitter );
    } else if (fitter.compare("KF5ParamsComb")==0) {
	return new KFParamsComb(settings, 5, fitter );
    } else if (fitter.compare("SimpleLR")==0) {
      return new SimpleLR(settings);
#ifdef USE_HLS
    } else if (fitter.compare("KF4ParamsCombHLS")==0){
      return new KFParamsCombCallHLS(settings, 4, fitter );
    } else if (fitter.compare("KF5ParamsCombHLS")==0){
      return new KFParamsCombCallHLS(settings, 5, fitter );
#endif
    } else {
      throw cms::Exception("TrackFitGeneric: ERROR you requested unknown track fitter")<<fitter<<endl;
    }
} 

}
