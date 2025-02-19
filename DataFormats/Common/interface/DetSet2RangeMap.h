#ifndef DataFormats_Common_DetSet2RangeMap_h
#define DataFormats_Common_DetSet2RangeMap_h

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/RangeMap.h"
// #include "DataFormats/Common/interface/DetSetAlgorithm.h"

#include <algorithm>

//FIXME remove New when ready
namespace edmNew {

  namespace dstvdetails {
    // copy from DS to RM
    template<typename B>
    struct ToRM {
      ToRM(edm::RangeMap<det_id_type, edm::OwnVector<B> > & irm) : rm(&irm){}
      edm::RangeMap<det_id_type, edm::OwnVector<B> > * rm;
      template<typename T>
      void operator()(edmNew::DetSet<T> const&  ds) {
	// make it easy
	// std::vector<T const *> v(ds.size());
	//std::transform(ds.begin(),ds.end(),v.begin(),dstvdetails::Pointer());
	if (!ds.empty()) rm->put(ds.id(), ds.begin(), ds.end());
      }
    };
  }

  // copy from DSTV to RangeMap
  template<typename T, typename B>
  void copy(DetSetVector<T> const&  dstv,
       edm::RangeMap<det_id_type, edm::OwnVector<B> > & rm) {
    dstvdetails::ToRM<B> torm(rm);
    std::for_each(dstv.begin(), dstv.end(), torm);
  }

}
 
#endif // DataFormats_Common_DetSet2RangeMap_h
