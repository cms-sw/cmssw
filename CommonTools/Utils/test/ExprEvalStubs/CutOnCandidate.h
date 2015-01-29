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
    template<typename F>
    void mask(Collection const& cands, Mask& mask, F f) const {
      mask.resize(cands.size()); 
      std::transform(cands.begin(),cands.end(),mask.begin(), [&](Collection::value_type const & c){ return f(*c);});
    }
    virtual void eval(Collection const&, Mask&) const = 0;
  };

  struct SelectCandidateCollection {
    using Collection = std::vector<reco::LeafCandidate const *>;
    template<typename F>
    void select(Collection& cands, F f) const {
      cands.erase(std::remove_if(cands.begin(),cands.end(),[&](Collection::value_type const &c){return !f(*c);}),cands.end());
    }
    virtual void eval(Collection&) const = 0;
  };


}
