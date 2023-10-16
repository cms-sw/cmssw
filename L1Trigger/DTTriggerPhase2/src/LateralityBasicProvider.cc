#include "L1Trigger/DTTriggerPhase2/interface/LateralityBasicProvider.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;
// ============================================================================
// Constructors and destructor
// ============================================================================
LateralityBasicProvider::LateralityBasicProvider(const ParameterSet &pset, edm::ConsumesCollector &iC)
    : LateralityProvider(pset, iC), debug_(pset.getUntrackedParameter<bool>("debug")) {
  if (debug_)
    LogDebug("LateralityBasicProvider") << "LateralityBasicProvider: constructor";

  fill_lat_combinations();
}

LateralityBasicProvider::~LateralityBasicProvider() {
  if (debug_)
    LogDebug("LateralityBasicProvider") << "LateralityBasicProvider: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void LateralityBasicProvider::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("LateralityBasicProvider") << "LateralityBasicProvider::initialiase";
}

void LateralityBasicProvider::run(edm::Event &iEvent,
                                  const edm::EventSetup &iEventSetup,
                                  MuonPathPtrs &muonpaths,
                                  std::vector<lat_vector> &lateralities) {
  if (debug_)
    LogDebug("LateralityBasicProvider") << "LateralityBasicProvider: run";

  // fit per SL (need to allow for multiple outputs for a single mpath)
  for (auto &muonpath : muonpaths) {
    analyze(muonpath, lateralities);
  }
}

void LateralityBasicProvider::finish() {
  if (debug_)
    LogDebug("LateralityBasicProvider") << "LateralityBasicProvider: finish";
};

//------------------------------------------------------------------
//--- Metodos privados
//------------------------------------------------------------------

void LateralityBasicProvider::analyze(MuonPathPtr &inMPath, std::vector<lat_vector> &lateralities) {
  if (debug_)
    LogDebug("LateralityBasicProvider") << "DTp2:analyze \t\t\t\t starts";
  for (auto &lat_combination : lat_combinations) {
    if (inMPath->missingLayer() == lat_combination.missing_layer &&
        inMPath->cellLayout()[0] == lat_combination.cellLayout[0] &&
        inMPath->cellLayout()[1] == lat_combination.cellLayout[1] &&
        inMPath->cellLayout()[2] == lat_combination.cellLayout[2] &&
        inMPath->cellLayout()[3] == lat_combination.cellLayout[3]) {
      lateralities.push_back(lat_combination.latcombs);
      return;
    }
  }
  lateralities.push_back(LAT_VECTOR_NULL);
  return;
}

void LateralityBasicProvider::fill_lat_combinations() {
  lat_combinations.push_back({-1, {0, 0, 0, -1}, {{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 1, 1}, {0, 0, 0, 0}}});
  lat_combinations.push_back({-1, {0, 0, 1, -1}, {{0, 0, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({-1, {0, 1, 0, -1}, {{0, 1, 0, 0}, {0, 1, 0, 1}, {1, 1, 0, 0}, {1, 1, 0, 1}}});
  lat_combinations.push_back({-1, {0, 1, 1, -1}, {{0, 1, 0, 0}, {0, 1, 1, 0}, {0, 1, 1, 1}, {0, 0, 0, 0}}});
  lat_combinations.push_back({-1, {1, 0, 0, -1}, {{1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 1}, {0, 0, 0, 0}}});
  lat_combinations.push_back({-1, {1, 0, 1, -1}, {{0, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 1, 0}, {1, 0, 1, 1}}});
  lat_combinations.push_back({-1, {1, 1, 0, -1}, {{0, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 0, 1}, {0, 0, 0, 0}}});
  lat_combinations.push_back({-1, {1, 1, 1, -1}, {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 1, 1, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({0, {0, 0, 0, -1}, {{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({0, {0, 0, 1, -1}, {{0, 0, 1, 0}, {0, 0, 1, 1}, {0, 1, 1, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({0, {0, 1, 0, -1}, {{0, 0, 0, 1}, {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 0}}});
  lat_combinations.push_back({0, {0, 1, 1, -1}, {{0, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({1, {0, 0, 0, -1}, {{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({1, {0, 0, 1, -1}, {{0, 0, 1, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({1, {0, 1, 0, -1}, {{0, 0, 0, 1}, {1, 0, 0, 0}, {1, 0, 0, 1}, {0, 0, 0, 0}}});
  lat_combinations.push_back({1, {0, 1, 1, -1}, {{0, 0, 1, 0}, {0, 0, 1, 1}, {1, 0, 1, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({1, {1, 1, 0, -1}, {{0, 0, 0, 1}, {1, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({1, {1, 1, 1, -1}, {{1, 0, 0, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({2, {0, 0, 0, -1}, {{0, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({2, {0, 0, 1, -1}, {{0, 1, 0, 0}, {0, 1, 0, 1}, {1, 1, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({2, {0, 1, 1, -1}, {{0, 1, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({2, {1, 0, 0, -1}, {{1, 0, 0, 0}, {1, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({2, {1, 0, 1, -1}, {{0, 0, 0, 1}, {1, 0, 0, 0}, {1, 0, 0, 1}, {0, 0, 0, 0}}});
  lat_combinations.push_back({2, {1, 1, 1, -1}, {{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({3, {0, 0, 0, -1}, {{0, 0, 1, 0}, {0, 1, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({3, {0, 1, 0, -1}, {{0, 1, 0, 0}, {0, 1, 1, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({3, {1, 0, 0, -1}, {{0, 0, 1, 0}, {1, 0, 0, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}}});
  lat_combinations.push_back({3, {1, 1, 0, -1}, {{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}});
};
