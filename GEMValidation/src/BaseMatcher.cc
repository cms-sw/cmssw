#include "BaseMatcher.h"

using namespace std;


BaseMatcher::BaseMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es)
: trk_(t), vtx_(v), conf_(ps), ev_(ev), es_(es), verbose_(0)
{
  // list of CSC chamber type numbers to use
  vector<int> csc_types = conf().getUntrackedParameter<vector<int> >("useCSCChamberTypes", vector<int>() );
  for (int i=0; i <= CSC_ME42; ++i) useCSCChamberTypes_[i] = false;
  for (auto t: csc_types)
  {
    if (t >= 0 && t <= CSC_ME42) useCSCChamberTypes_[t] = 1;
  }
  // empty list means use all the chamber types
  if (csc_types.empty()) useCSCChamberTypes_[CSC_ALL] = 1;
}


BaseMatcher::~BaseMatcher() {}


bool BaseMatcher::useCSCChamberType(int csc_type)
{
  if (csc_type < 0 || csc_type > CSC_ME42) return false;
  return useCSCChamberTypes_[csc_type];
}
