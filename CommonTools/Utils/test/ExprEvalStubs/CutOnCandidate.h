#include <vector>
#include <algorithm>
#include <numeric>
#include<memory>


#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace eetest {
  struct CutOnCandidate {
    virtual bool eval(reco::LeafCandidate const&) const = 0;
  };

  struct ValueOnCandidate {
    virtual double eval(reco::LeafCandidate const&) const =0;
  };

  struct MaskCandidateCollection {
    using Collection = std::vector<reco::LeafCandidate const *>;
    using Mask = std::vector<bool>;
    virtual void eval(Collection const&, Mask&) const = 0;
  };

  struct SelectCandidateCollection {
    using Collection = std::vector<reco::LeafCandidate const *>;
    using Indices = std::vector<unsigned int>;
    virtual void eval(Collection const&, Indices&) const = 0;
  };


}
