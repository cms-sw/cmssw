#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPFilter::MPFilter(const ParameterSet& pset) {
  // Obtention of parameters
  debug_ = pset.getUntrackedParameter<bool>("debug");
}

MPFilter::~MPFilter() {}
