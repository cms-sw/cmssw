#ifndef L1Trigger_TrackFindingTMTT_GlobalCacheTMTT_h
#define L1Trigger_TrackFindingTMTT_GlobalCacheTMTT_h

// Data shared across all streams by TMTT L1 tracking.
//
// Provides the python configuration parameters
// & optional histogramming/debugging.

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/HTrphi.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubWindowSuggest.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"
#include "L1Trigger/TrackFindingTMTT/interface/Histos.h"

#include <list>
#include <memory>

namespace tmtt {

  class GlobalCacheTMTT {
  public:
    GlobalCacheTMTT(const edm::ParameterSet& iConfig)
        : settings_(iConfig),              // Python configuration params
          htRphiErrMon_(),                 // rphi HT error monitoring
          stubWindowSuggest_(&settings_),  // Recommend FE stub window sizes.
          hists_(&settings_)               // Histograms
    {
      hists_.book();
    }

    // Get functions
    const Settings& settings() const { return settings_; }
    HTrphi::ErrorMonitor& htRphiErrMon() const { return htRphiErrMon_; }
    StubWindowSuggest& stubWindowSuggest() const { return stubWindowSuggest_; }
    const std::list<TrackerModule>& listTrackerModule() const { return listTrackerModule_; }
    Histos& hists() const { return hists_; }

    // Set functions
    void setListTrackerModule(const std::list<TrackerModule>& list) const {
      // Allow only one thread to run this function at a time
      static std::mutex myMutex;
      std::lock_guard<std::mutex> myGuard(myMutex);

      // Only need one copy of tracker geometry for histogramming.
      if (listTrackerModule_.empty())
        listTrackerModule_ = list;
    }

  private:
    Settings settings_;
    mutable HTrphi::ErrorMonitor htRphiErrMon_;
    mutable StubWindowSuggest stubWindowSuggest_;
    mutable std::list<TrackerModule> listTrackerModule_;
    mutable Histos hists_;
  };

}  // namespace tmtt

#endif
