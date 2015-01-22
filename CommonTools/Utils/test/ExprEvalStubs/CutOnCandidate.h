#include <vector>
#include <algorithm>
#include <numeric>
#include<memory>


#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace eetest {
  struct CutOnCandidate {
    virtual bool eval(reco::Candidate const&) const = 0;
  };

  struct ValueOnCandidate {
    virtual double eval(reco::Candidate const&) const =0;
  };

}
