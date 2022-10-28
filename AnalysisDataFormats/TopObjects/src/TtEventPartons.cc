#include "AnalysisDataFormats/TopObjects/interface/TtEventPartons.h"

void TtEventPartons::expand(std::vector<int>& vec) const {
  std::vector<int>::iterator vecIter = vec.begin();
  for (unsigned i = 0; i < ignorePartons_.size(); i++) {
    if (ignorePartons_[i]) {
      vecIter = vec.insert(vecIter, -3);
    }
    ++vecIter;
  }
}

void TtEventPartons::prune(std::vector<const reco::Candidate*>& vec) const {
  unsigned int nIgnoredPartons = 0;
  for (unsigned i = 0; i < ignorePartons_.size(); i++) {
    if (ignorePartons_[i]) {
      vec.erase(vec.begin() + (i - nIgnoredPartons));
      nIgnoredPartons++;
    }
  }
}
