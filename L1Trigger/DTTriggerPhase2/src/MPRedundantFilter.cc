#include <memory>
#include "L1Trigger/DTTriggerPhase2/interface/MPRedundantFilter.h"

using namespace edm;
using namespace std;
using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPRedundantFilter::MPRedundantFilter(const ParameterSet& pset)
    : MPFilter(pset), debug_(pset.getUntrackedParameter<bool>("debug")), maxBufferSize_(8) {}

MPRedundantFilter::~MPRedundantFilter() {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPRedundantFilter::initialise(const edm::EventSetup& iEventSetup) { buffer_.clear(); }

void MPRedundantFilter::run(edm::Event& iEvent,
                            const edm::EventSetup& iEventSetup,
                            MuonPathPtrs& inMPaths,
                            MuonPathPtrs& outMPaths) {
  buffer_.clear();
  for (auto muonpath = inMPaths.begin(); muonpath != inMPaths.end(); ++muonpath) {
    filter(*muonpath, outMPaths);
  }
  buffer_.clear();
}

void MPRedundantFilter::filter(MuonPathPtr& mPath, MuonPathPtrs& outMPaths) {
  if (mPath == nullptr)
    return;

  if (!isInBuffer(mPath)) {
    // Remove the first element (the oldest)
    if (buffer_.size() == maxBufferSize_)
      buffer_.pop_front();
    // Insert last path as new element
    buffer_.push_back(mPath);

    // Send a copy
    auto mpAux = std::make_shared<MuonPath>(mPath);
    outMPaths.push_back(mpAux);
  }
}

bool MPRedundantFilter::isInBuffer(MuonPathPtr& mPath) {
  bool ans = false;

  if (!buffer_.empty()) {
    for (unsigned int i = 0; i < buffer_.size(); i++)
      if (mPath->isEqualTo((MuonPath*)buffer_.at(i).get())) {
        ans = true;
        break;
      }
  }
  return ans;
}
