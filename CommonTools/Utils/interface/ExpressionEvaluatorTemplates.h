#ifndef CommonToolsUtilsExpressionEvaluatorTemplates_H
#define	CommonToolsUtilsExpressionEvaluatorTemplates_H
#include <vector>
#include <algorithm>
#include<numeric>
#include<limits>
#include<memory>
#include<tuple>

namespace reco {

  template<typename Ret, typename... Args>
  struct genericExpression {
    virtual Ret operator()(Args ...) const =0;
  };


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

  template<typename Object>
  struct SelectIndecesInCollection {
    using Collection = std::vector<Object const *>;
    using Indices = std::vector<unsigned int>;
    template<typename F>
    void select(Collection const & cands, Indices& inds, F f) const {
      unsigned int i=0;
      for (auto const & c : cands) { if(f(*c)) inds.push_back(i); ++i; }
    }
    virtual void eval(Collection const&, Indices&) const = 0;
  };


}

#endif	// CommonToolsUtilsExpressionEvaluatorTemplates_H

