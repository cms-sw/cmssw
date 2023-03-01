#ifndef __STUBKILLER_H__
#define __STUBKILLER_H__

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "TRandom3.h"
#include "TMath.h"

using namespace std;

class StubKiller {
public:
  StubKiller();
  ~StubKiller() {}

  void initialise(unsigned int killScenario,
                  const TrackerTopology* trackerTopology,
                  const TrackerGeometry* trackerGeometry);

  bool killStub(const TTStub<Ref_Phase2TrackerDigi_>* stub,
                const vector<int> layersToKill,
                const double minPhiToKill,
                const double maxPhiToKill,
                const double minZToKill,
                const double maxZToKill,
                const double minRToKill,
                const double maxRToKill,
                const double fractionOfStubsToKillInLayers,
                const double fractionOfStubsToKillEverywhere);

  bool killStub(const TTStub<Ref_Phase2TrackerDigi_>* stub);

  bool killStubInDeadModule(const TTStub<Ref_Phase2TrackerDigi_>* stub);

  map<DetId, float> getListOfDeadModules() { return deadModules_; }

private:
  void chooseModulesToKill();
  void addDeadLayerModulesToDeadModuleList();

  unsigned int killScenario_;
  const TrackerTopology* trackerTopology_;
  const TrackerGeometry* trackerGeometry_;

  vector<int> layersToKill_;
  double minPhiToKill_;
  double maxPhiToKill_;
  double minZToKill_;
  double maxZToKill_;
  double minRToKill_;
  double maxRToKill_;
  double fractionOfStubsToKillInLayers_;
  double fractionOfStubsToKillEverywhere_;
  double fractionOfModulesToKillEverywhere_;

  map<DetId, float> deadModules_;
};

#endif
