#include "L1Trigger/DTTriggerPhase2/interface/MPCleanHitsFilter.h"

using namespace edm;
using namespace std;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPCleanHitsFilter::MPCleanHitsFilter(const ParameterSet &pset)
    : MPFilter(pset), debug_(pset.getUntrackedParameter<bool>("debug")) {
  timeTolerance_ = pset.getParameter<int>("timeTolerance");
  // probably something close to the max time drift (400ns/2) is a reasonable value
}
void MPCleanHitsFilter::run(edm::Event &iEvent,
                            const edm::EventSetup &iEventSetup,
                            MuonPathPtrs &inMPaths,
                            MuonPathPtrs &outMPaths) {
  for (const auto &mpath : inMPaths) {
    auto mpAux = std::make_shared<MuonPath>(*mpath);
    removeOutliers(mpAux);  // remove hits that are more than 1 bX from the meantime.

    outMPaths.emplace_back(mpAux);
  }
}

void MPCleanHitsFilter::removeOutliers(MuonPathPtr &mpath) {
  int MeanTime = getMeanTime(mpath);
  for (int i = 0; i < mpath->nprimitives(); i++) {
    if (!mpath->primitive(i)->isValidTime())
      continue;
    if (std::abs(mpath->primitive(i)->tdcTimeStamp() - MeanTime) > timeTolerance_) {
      mpath->primitive(i)->setTDCTimeStamp(-1);  //invalidate hit
      mpath->primitive(i)->setChannelId(-1);     //invalidate hit
    }
  }
}

double MPCleanHitsFilter::getMeanTime(MuonPathPtr &mpath) {
  float meantime = 0.;
  float count = 0.;
  for (int i = 0; i < mpath->nprimitives(); i++) {
    if (mpath->primitive(i) == nullptr)
      continue;
    if (!mpath->primitive(i)->isValidTime())
      continue;
    meantime += mpath->primitive(i)->tdcTimeStamp();
    count++;
  }
  return meantime / count;
}
