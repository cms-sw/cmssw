///=== Create requested track fitter

#include "L1Trigger/TrackFindingTMTT/interface/TrackFitFactory.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/ChiSquaredFit4.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/SimpleLR4.h"
#ifdef USE_HLS
#include "L1Trigger/TrackFindingTMTT/interface/HLS/KFParamsCombCallHLS.h"
#endif
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

namespace tmtt {

  namespace trackFitFactory {

    std::unique_ptr<TrackFitGeneric> create(const std::string& fitterName, const Settings* settings) {
      if (fitterName == "ChiSquaredFit4") {
        return std::make_unique<ChiSquaredFit4>(settings, 4);
      } else if (fitterName == "KF4ParamsComb") {
        return std::make_unique<KFParamsComb>(settings, 4, fitterName);
      } else if (fitterName == "KF5ParamsComb") {
        return std::make_unique<KFParamsComb>(settings, 5, fitterName);
      } else if (fitterName == "SimpleLR4") {
        return std::make_unique<SimpleLR4>(settings);
#ifdef USE_HLS
      } else if (fitterName == "KF4ParamsCombHLS") {
        return std::make_unique<KFParamsCombCallHLS>(settings, 4, fitterName);
      } else if (fitterName == "KF5ParamsCombHLS") {
        return std::make_unique<KFParamsCombCallHLS>(settings, 5, fitterName);
#endif
      } else {
        throw cms::Exception("BadConfig")
            << "TrackFitFactory: ERROR you requested unknown track fitterName: " << fitterName;
      }
    }

  }  // namespace trackFitFactory

}  // namespace tmtt
