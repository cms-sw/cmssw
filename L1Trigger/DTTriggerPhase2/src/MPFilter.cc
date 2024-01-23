#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPFilter::MPFilter(const ParameterSet& pset) : debug_(pset.getUntrackedParameter<bool>("debug")) {
  // Obtention of parameters
  int wh, st, se, maxdrift;
  maxdrift_filename_ = pset.getParameter<edm::FileInPath>("maxdrift_filename");
  std::ifstream ifind(maxdrift_filename_.fullPath());
  if (ifind.fail()) {
    throw cms::Exception("Missing Input File")
        << "MPSLFilter::MPSLFilter() -  Cannot find " << maxdrift_filename_.fullPath();
  }
  while (ifind.good()) {
    ifind >> wh >> st >> se >> maxdrift;
    maxdriftinfo_[wh][st][se] = maxdrift;
  }
}

MPFilter::~MPFilter() {}
