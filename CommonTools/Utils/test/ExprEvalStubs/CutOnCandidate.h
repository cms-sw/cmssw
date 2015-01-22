#include <vector>
#include <algorithm>
#include <numeric>
#include<memory>


#include "DataFormats/Candidate/interface/Candidate.h"

namespace eetest {
  struct CutOnCandidate {
    virtual bool eval(reco::Candidate const&)=0;
  };
}
