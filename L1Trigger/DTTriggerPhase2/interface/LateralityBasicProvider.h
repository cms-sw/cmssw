#ifndef L1Trigger_DTTriggerPhase2_LateralityBasicProvider_h
#define L1Trigger_DTTriggerPhase2_LateralityBasicProvider_h

#include "L1Trigger/DTTriggerPhase2/interface/LateralityProvider.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

struct lat_combination {
  short missing_layer;
  short cellLayout[cmsdt::NUM_LAYERS];
  lat_vector latcombs;
};

// ===============================================================================
// Class declarations
// ===============================================================================

class LateralityBasicProvider : public LateralityProvider {
public:
  // Constructors and destructor
  LateralityBasicProvider(const edm::ParameterSet &pset, edm::ConsumesCollector &iC);
  ~LateralityBasicProvider() override;

  // Main methods
  void initialise(const edm::EventSetup &iEventSetup) override;
  void run(edm::Event &iEvent,
           const edm::EventSetup &iEventSetup,
           MuonPathPtrs &inMpath,
           std::vector<lat_vector> &lateralities) override;

  void finish() override;

  // Other public methods

private:
  // Private methods
  void analyze(MuonPathPtr &inMPath, std::vector<lat_vector> &lateralities);
  void fill_lat_combinations();
  // Private attributes
  const bool debug_;
  std::vector<lat_combination> lat_combinations;
};

#endif
