#ifndef MTDStablePhiSort_H
#define MTDStablePhiSort_H
// #include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

#include <utility>
#include <vector>
#include <algorithm>

#include <cmath>

#include<boost/bind.hpp>

namespace details {
  template<typename Object, typename Scalar> 
  struct PhiSortElement {
    typedef PhiSortElement<Object,Scalar> self;
    
    template<typename Extractor>
    static self 
    build(Object & o, Extractor const & extr) { 
      return self(&o, extr(o));
    }
    
    PhiSortElement()
      : pointer(nullptr) {}
    PhiSortElement(Object * p, Scalar v):
      pointer(p),
      value(v) {}
    
    Object * pointer;
    Scalar value;
    
    bool operator<(self const & rh) const {
      return value<rh.value;
    }
    Object const &
    obj() const { return *pointer;}
  };
  
  
}

template<typename RandomAccessIterator, typename Extractor>
void MTDStablePhiSort(RandomAccessIterator begin,
			  RandomAccessIterator end,
			  const Extractor& extr) {
  
  typedef typename Extractor::result_type        Scalar;
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;
  
  typedef details::PhiSortElement<value_type, Scalar> Element;
  
  
  
  std::vector<Element> tmpvec(end-begin);
  std::transform(begin,end,tmpvec.begin(),
		 boost::bind(Element::template build<Extractor>,_1, boost::cref(extr))
		 );
  
  std::vector<Element> tmpcop(end-begin);
  
  std::sort(tmpvec.begin(), tmpvec.end());
  
  const unsigned int vecSize = tmpvec.size();
  
  
  
  // special tratment of the TEC modules of rings in petals near phi=0
  // there are at most 5 modules, no other structure has less than ~10 elements to order in phi
  // hence the special case in phi~0 if the size of the elements to order is <=5
  const unsigned int nMaxModulesPerRing = 5;
  bool phiZeroCase = true;
  //
  const double phiMin = M_PI_4;
  const double phiMax = 2*M_PI-phiMin;
  //
  if( vecSize > nMaxModulesPerRing ) {
    //stability check
    // check if the last element is too near to zero --> probably it is zero
    double tolerance = 0.000001;
    if( fabs(tmpvec.back().value - 0) < tolerance       // near 0
	||
	fabs(tmpvec.back().value - 2*M_PI) < tolerance ) { // near 2pi
      // move it to front 
      tmpvec.insert(tmpvec.begin(),tmpvec.back());
      tmpvec.pop_back();
    }
  }
  else {
    // check if all the elements have phi<phiMin or phi>phiMax to be sure we are near phi~0 (angles are in [0,2pi) range)
    // if a phi goes out from [0,phiMin]U[phiMax,2pi) it is not the case
    // sorted. if first > phiMax all other will also...
    typename std::vector<Element>::iterator p = 
      std::find_if(tmpvec.begin(),tmpvec.end(), boost::bind(&Element::value,_1) > phiMin);
    phiZeroCase = !(p!=tmpvec.end() && (*p).value<phiMax);
    
    // go on if this is the petal phi~0 case, restricted to the case where all the |phi| are in range [0,phiMin]
    if(phiZeroCase) {
      // in this case the ordering must be: ('negative' values, >) and then ('positive' values, >) in (-pi,pi] mapping
      // already sorted, just swap ranges
      if(p!=tmpvec.end()) {
	tmpvec.insert(tmpvec.begin(),p,tmpvec.end());
	tmpvec.resize(vecSize);
      }
    }
  }
  
  // overwrite the input range with the sorted values
  // copy of input container not necessary, but tricky to avoid
  std::vector<value_type> tmpvecy(vecSize);
  std::transform(tmpvec.begin(),tmpvec.end(),tmpvecy.begin(),boost::bind(&Element::obj,_1));
  std::copy(tmpvecy.begin(),tmpvecy.end(),begin);
  
}

#endif

