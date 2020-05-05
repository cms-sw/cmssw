#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPFilter::MPFilter(const ParameterSet& pset) {
  // Obtention of parameters
  debug = pset.getUntrackedParameter<bool>("debug");
  if (debug)
    cout << "MPFilter: constructor" << endl;
}

MPFilter::~MPFilter() {
  if (debug)
    cout << "MPFilter: destructor" << endl;
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
