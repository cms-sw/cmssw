#ifndef L1Trigger_DTTriggerPhase2_LateralityCoarsedProvider_h
#define L1Trigger_DTTriggerPhase2_LateralityCoarsedProvider_h

#include "L1Trigger/DTTriggerPhase2/interface/LateralityProvider.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

struct lat_coarsed_combination {
  short missing_layer;
  short cellLayout[cmsdt::NUM_LAYERS];
  short coarsed_times[cmsdt::NUM_LAYERS];
  lat_vector latcombs;
};

// ===============================================================================
// Class declarations
// ===============================================================================

class LateralityCoarsedProvider : public LateralityProvider {
public:
  // Constructors and destructor
  LateralityCoarsedProvider(const edm::ParameterSet &pset, edm::ConsumesCollector &iC);
  ~LateralityCoarsedProvider() override;

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
  std::vector<short> coarsify_times(MuonPathPtr &inMPath);
  void fill_lat_combinations();
  // Private attributes
  const bool debug_;
  std::vector<lat_coarsed_combination> lat_combinations;
};

#endif
