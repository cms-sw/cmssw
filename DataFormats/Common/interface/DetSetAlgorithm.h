#ifndef DataFormats_Common_DetSetAlgorithm_h
#define DataFormats_Common_DetSetAlgorithm_h

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include <algorithm>
#include <functional>

//FIXME remove New when ready
namespace edmNew {

  // adapt the RecHit accessors to DSTV
  template <typename DSTV, typename A, typename B>
  typename DSTV::Range detsetRangeFromPair(DSTV const & v, std::pair<A,B> const & p) {
    return v.equal_range(p.first,p.second);
  }


  // invoke f for each object in the range of DataSets selected by sel
  // to ease use, f is passed by reference
  template <typename DSTV, typename A, typename B, typename F>
  void foreachDetSetObject(DSTV const & v, std::pair<A,B> const & sel, F & f) {
    typedef typename DSTV::data_type data_type;
    typename DSTV::Range range = detsetRangeFromPair(v,sel);
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++)
      std::for_each((*id).begin(), (*id).end(),
		    std::function<void(const data_type &)>(std::ref(f)));
  }

  namespace dstvdetails {

    struct Pointer {
      template<typename H> 
      H const * operator()(H const& h) const { return &h;}
    };

  }

  // to write an easy iterator on objects in a range of DataSets selected by sel
  // is left as exercise to the reader 
  // here we provide this not optimal solution...
  template <typename DSTV, typename A, typename B, typename T>
  void copyDetSetRange(DSTV const & dstv,    
		       std::vector<T const *> & v, 
		       std::pair<A,B> const & sel) {
    typename DSTV::Range range = dstv.equal_range(sel.first,sel.second);
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++){
      size_t cs = v.size();
      v.resize(cs+(*id).size());
      std::transform((*id).begin(), (*id).end(),v.begin()+cs,dstvdetails::Pointer());
    } 
  }
}


#endif //  DataFormats_Common_DetSetAlgorithm_h
