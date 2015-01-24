#ifndef CommonToolsUtilsExpressionEvaluatorTemplates_H
#define	CommonToolsUtilsExpressionEvaluatorTemplates_H
#include <vector>
#include <algorithm>

namespace reco {
  template<typename Object>
  struct CutOnObject {
    virtual bool eval(Object const&) const = 0;
  };

  template<typename Object>
  struct ValueOnObject {
    virtual double eval(Object const&) const =0;
  };

  template<typename Object>
  struct MaskCollection {
    using Collection = std::vector<Object const *>;
    using Mask = std::vector<bool>;
    template<typename F>
    void mask(Collection const& cands, Mask& mask, F f) const {
      mask.resize(cands.size()); 
      std::transform(cands.begin(),cands.end(),mask.begin(), [&](typename Collection::value_type const & c){ return f(*c);});
    }
    virtual void eval(Collection const&, Mask&) const = 0;
  };

  template<typename Object>
  struct SelectInCollection {
    using Collection = std::vector<Object const *>;
    template<typename F>
    void select(Collection& cands, F f) const {
      cands.erase(std::remove_if(cands.begin(),cands.end(),[&](typename Collection::value_type const &c){return !f(*c);}),cands.end());
    }
    virtual void eval(Collection&) const = 0;
  };

}

#endif	// CommonToolsUtilsExpressionEvaluatorTemplates_H

