#ifndef L1Trigger_TrackFindingTMTT_StubKiller_h
#define L1Trigger_TrackFindingTMTT_StubKiller_h

// Kill some stubs to emulate dead tracker modules.
// Author: Emyr Clement (2018)
// Tidy up: Ian Tomalin (2020)

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandomEngine.h"

namespace tmtt {

  class StubKiller {
  public:
    enum class KillOptions { none = 0, layer5 = 1, layer1 = 2, layer1layer2 = 3, layer1disk1 = 4, random = 5 };

    StubKiller(KillOptions killScenario,
               const TrackerTopology* trackerTopology,
               const TrackerGeometry* trackerGeometry,
               const edm::Event& iEvent);

    // Indicate if given stub was killed by dead tracker module, based on dead module scenario.
    bool killStub(const TTStub<Ref_Phase2TrackerDigi_>* stub) const;

    // Indicate if given stub was killed by dead tracker module, based on dead regions specified here,
    // and ignoring dead module scenario.
    bool killStub(const TTStub<Ref_Phase2TrackerDigi_>* stub,
                  const std::vector<int>& layersToKill,
                  const double minPhiToKill,
                  const double maxPhiToKill,
                  const double minZToKill,
                  const double maxZToKill,
                  const double minRToKill,
                  const double maxRToKill,
                  const double fractionOfStubsToKillInLayers,
                  const double fractionOfStubsToKillEverywhere) const;

    // Indicate if given stub was in (partially) dead tracker module, based on dead module scenario.
    bool killStubInDeadModule(const TTStub<Ref_Phase2TrackerDigi_>* stub) const;

    // List of all modules declared as (partially) dead, with fractional deadness of each.
    const std::map<DetId, float>& listOfDeadModules() const { return deadModules_; }

  private:
    // Identify modules to be killed, chosen randomly from those in the whole tracker.
    void chooseModulesToKill();
    //  Identify modules to be killed, chosen based on location in tracker.
    void addDeadLayerModulesToDeadModuleList();

    KillOptions killScenario_;
    const TrackerTopology* trackerTopology_;
    const TrackerGeometry* trackerGeometry_;

    std::vector<int> layersToKill_;
    double minPhiToKill_;
    double maxPhiToKill_;
    double minZToKill_;
    double maxZToKill_;
    double minRToKill_;
    double maxRToKill_;
    double fractionOfStubsToKillInLayers_;
    double fractionOfStubsToKillEverywhere_;
    double fractionOfModulesToKillEverywhere_;

    std::map<DetId, float> deadModules_;

    edm::Service<edm::RandomNumberGenerator> rndmService_;
    CLHEP::HepRandomEngine* rndmEngine_;
  };

};  // namespace tmtt

#endif
