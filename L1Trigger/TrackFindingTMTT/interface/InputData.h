#ifndef L1Trigger_TrackFindingTMTT_InputData_h
#define L1Trigger_TrackFindingTMTT_InputData_h

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <list>

namespace tmtt {

  class Settings;
  class StubWindowSuggest;
  class DegradeBend;

  //=== Unpacks stub & tracking particle (truth) data into user-friendlier format in Stub & TP classes.
  //=== Also makes B-field available to Settings class.

  class InputData {
  public:
    InputData(const edm::Event& iEvent,
              const edm::EventSetup& iSetup,
              const Settings* settings,
              StubWindowSuggest* stubWindowSuggest,
              const DegradeBend* degradeBend,
              const TrackerGeometry* trackerGeometry,
              const TrackerTopology* trackerTopology,
              const std::list<TrackerModule>& listTrackerModule,
              const edm::EDGetTokenT<TrackingParticleCollection> tpToken,
              const edm::EDGetTokenT<TTStubDetSetVec> stubToken,
              const edm::EDGetTokenT<TTStubAssMap> stubTruthToken,
              const edm::EDGetTokenT<TTClusterAssMap> clusterTruthToken,
              const edm::EDGetTokenT<reco::GenJetCollection> genJetToken);

    // Info about each tracker module
    const std::list<TrackerModule>& trackerModules() const { return trackerModules_; };

    // Get tracking particles
    const std::list<TP>& getTPs() const { return vTPs_; }
    // Get stubs that would be output by the front-end readout electronics
    const std::list<Stub*>& stubs() const { return vStubs_; }
    // Ditto but const
    const std::list<const Stub*>& stubsConst() const { return vStubsConst_; }

    //--- of minor importance ...

    // Get number of stubs prior to applying tighted front-end readout electronics cuts specified in section StubCuts of Analyze_Defaults_cfi.py. (Only used to measure the efficiency of these cuts).
    const std::list<Stub>& allStubs() const { return vAllStubs_; }

  private:
    bool enableMCtruth_;  // Notes if job will use MC truth info.

    std::list<TrackerModule> trackerModules_;  // Info about each tracker module.

    std::list<TP> vTPs_;                  // tracking particles
    std::list<Stub*> vStubs_;             // stubs that would be output by the front-end readout electronics.
    std::list<const Stub*> vStubsConst_;  // ditto but const

    //--- Used for a few minor studies ...

    // all stubs, even those that would fail any tightened front-end readout electronic cuts specified in section StubCuts of Analyze_Defaults_cfi.py. (Only used to measure the efficiency of these cuts).
    std::list<Stub> vAllStubs_;

    // Recommends optimal FE stub window sizes.
    StubWindowSuggest* stubWindowSuggest_;
    // Degrades bend to allow for FE stub bend encoding.
    const DegradeBend* degradeBend_;
  };

}  // namespace tmtt
#endif
