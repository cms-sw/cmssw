#include "L1Trigger/DTTriggerPhase2/interface/LateralityProvider.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
LateralityProvider::LateralityProvider(const ParameterSet& pset, edm::ConsumesCollector& iC)
    : debug_(pset.getUntrackedParameter<bool>("debug")) {}

LateralityProvider::~LateralityProvider() {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void LateralityProvider::initialise(const edm::EventSetup& iEventSetup) {}

void LateralityProvider::finish(){};

void LateralityProvider::run(edm::Event& iEvent,
                             const edm::EventSetup& iEventSetup,
                             MuonPathPtrs& inMpath,
                             std::vector<lat_vector>& lateralities){};
