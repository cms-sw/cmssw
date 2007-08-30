#ifndef Common_DetSet2RangeMap_H
#define Common_DetSet2RangeMap_H

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/RangeMap.h"
// #include "DataFormats/Common/interface/DetSetAlgorithm.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <boost/ref.hpp>
// #include <boost/bind.hpp>
// #include <boost/function.hpp>
#include <algorithm>

//FIXME remove New when ready
namespace edmNew {

  namespace dstvdetails {
    // copy from DS to RM
    template<typename B>
    struct ToRM {
      ToRm(edm::RangeMap<DetId, edm::OwnVector<B> > & irm) : rm(irm){}
      edm::RangeMap<DetId, edm::OwnVector<B> > & rm;
      template<typename T>
      void operator()(edmNew::DetSet<T> const&  ds) {
	// make it easy
	// std::vector<T const *> v(ds.size());
	//std::transform(ds.begin(),ds.end(),v.begin(),dstvdetails::Pointer());
	rm.put(ds.id(), ds.begin(), ds.end());
      }
    };
  }

  // copy from DSTV to RangeMap
  template<typename T, typename B>
  void copy(DetSetVector<T> const&  dstv,
       edm::RangeMap<DetId, edm::OwnVector<B> > & rm) {
    dstvdetails::ToRM<B> torm(rm);
    std::for_each(dstv.begin(), dstv.end(), boost::ref(torm));
  }

}
 
#endif Common_DetSet2RangeMap_H
