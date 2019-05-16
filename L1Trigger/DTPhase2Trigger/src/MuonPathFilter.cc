#include "L1Trigger/DTPhase2Trigger/interface/MuonPathFilter.h"

using namespace edm;
using namespace std;



// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathFilter::MuonPathFilter(const ParameterSet& pset) {
  // Obtention of parameters
  debug         = pset.getUntrackedParameter<Bool_t>("debug");
  if (debug) cout <<"MuonPathFilter: constructor" << endl;
}


MuonPathFilter::~MuonPathFilter() {
  if (debug) cout <<"MuonPathFilter: destructor" << endl;
}



// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
