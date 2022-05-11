#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzer.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAnalyzer::MuonPathAnalyzer(const ParameterSet& pset, edm::ConsumesCollector& iC)
    : debug_(pset.getUntrackedParameter<bool>("debug")) {}

MuonPathAnalyzer::~MuonPathAnalyzer() {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAnalyzer::initialise(const edm::EventSetup& iEventSetup) {}

void MuonPathAnalyzer::finish(){};
