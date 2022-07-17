#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
MotherGrouping::MotherGrouping(const ParameterSet& pset, edm::ConsumesCollector& iC)
    : debug_(pset.getUntrackedParameter<bool>("debug")) {}

MotherGrouping::~MotherGrouping() {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MotherGrouping::initialise(const edm::EventSetup& iEventSetup) {}

void MotherGrouping::run(Event& iEvent,
                         const EventSetup& iEventSetup,
                         const DTDigiCollection& digis,
                         MuonPathPtrs& mpaths) {}

void MotherGrouping::finish(){};
