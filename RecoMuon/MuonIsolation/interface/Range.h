#ifndef MuonIsolation_Range_H
#define MuonIsolation_Range_H

/** \class muonisolation::Range
 *  Define a range [aMin,aMax] 
 */

#include <iostream>
#include <utility>
#include <algorithm>

namespace muonisolation {

template<class T> class Range : public std::pair<T,T> {
public:

  Range() { }

  Range(const T & aMin, const T & aMax) : std::pair<T,T> (aMin,aMax) { }

  Range(const std::pair<T,T> & aPair ) : std::pair<T,T> (aPair) { }  

  const T & min() const { return this->first; }

  const T & max() const { return this->second; }

  T mean() const { return (this->first+this->second)/2.; }

  bool empty() const { return (this->second < this->first); }

  bool inside(const T & value) const {
    if (value < this->first || this->second < value)  return false; else  return true;
  }

  void sort() { if (empty() ) std::swap(this->first,this->second); }
};

}

template <class T> std::ostream & operator<<( std::ostream& out, const muonisolation::Range<T>& r) 
{ return out << "("<<r.min()<<","<<r.max()<<")"; }

#endif
