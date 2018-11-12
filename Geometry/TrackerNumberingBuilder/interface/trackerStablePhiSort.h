#ifndef Geometry_TrackerNumberingBuilder_trackerStablePhiSort_H
#define Geometry_TrackerNumberingBuilder_trackerStablePhiSort_H

#include <vector>
#include <algorithm>
#include <cmath>

template<class RandomAccessIterator, class Extractor>
void trackerStablePhiSort(RandomAccessIterator begin, RandomAccessIterator end, const Extractor& extr)
{
  using Scalar  = decltype(extr(*begin));
  using Value   = typename std::iterator_traits<RandomAccessIterator>::value_type;
  using Element = std::pair<Scalar, Value*>;

  std::vector<Element> tmpvec(end-begin);
  std::transform(begin,end,tmpvec.begin(), [&extr](Value& it){ return Element(extr(it), &it); });
  
  std::sort(tmpvec.begin(), tmpvec.end());
  
  // special tratment of the TEC modules of rings in petals near phi=0 there
  // are at most 5 modules, no other structure has less than ~10 elements to
  // order in phi hence the special case in phi~0 if the size of the elements
  // to order is <=5
  constexpr unsigned int nMaxModulesPerRing = 5;
  constexpr double       phiMin             = M_PI_4;
  constexpr double       phiMax             = 2 * M_PI - phiMin;
  constexpr double       tolerance          = 0.000001;

  const unsigned int n = tmpvec.size();

  if( n > nMaxModulesPerRing ) {
    // stability check
    // check if the last element is too near to zero --> probably it is zero
    if(    std::abs(tmpvec.back().first - 0)      < tolerance   // near 0
        || std::abs(tmpvec.back().first - 2*M_PI) < tolerance ) // near 2pi
    {
      // move it to front 
      tmpvec.insert(tmpvec.begin(),tmpvec.back());
      tmpvec.pop_back();
    }
  } else {
    // check if all the elements have phi<phiMin or phi>phiMax to be sure we
    // are near phi~0 (angles are in [0,2pi) range) if a phi goes out from
    // [0,phiMin]U[phiMax,2pi) it is not the case sorted. if first > phiMax all
    // other will also...
    auto p = std::find_if(tmpvec.begin(),tmpvec.end(), [&phiMin](auto const& x){ return x.first > phiMin; });
    
    // go on if this is the petal phi~0 case, restricted to the case where all
    // the |phi| are in range [0,phiMin]
    if(p == tmpvec.end() || p->first >= phiMax) {
      // in this case the ordering must be: ('negative' values, >) and then
      // ('positive' values, >) in (-pi,pi] mapping already sorted, just swap
      // ranges
      if(p!=tmpvec.end()) {
        tmpvec.insert(tmpvec.begin(),p,tmpvec.end());
        tmpvec.resize(n);
      }
    }
  }
  
  // overwrite the input range with the sorted values
  // copy of input container not necessary, but tricky to avoid
  std::vector<Value> tmpvecy(n);
  std::transform(tmpvec.begin(),tmpvec.end(),tmpvecy.begin(), [](auto& x){ return *x.second; });
  std::copy(tmpvecy.begin(),tmpvecy.end(),begin);
}

#endif
