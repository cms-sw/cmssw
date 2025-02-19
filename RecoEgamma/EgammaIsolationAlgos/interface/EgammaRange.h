#ifndef EgammaIsolationAlgos_EgammaRange_H
#define EgammaIsolationAlgos_EgammaRange_H

/** \class egammaisolation::EgammaRange
 *  Define a range [aMin,aMax] 
 */

#include <iostream>
#include <utility>
#include <algorithm>

namespace egammaisolation {

template<class T> class EgammaRange : public std::pair<T,T> {
public:

  EgammaRange() { }

  EgammaRange(const T & aMin, const T & aMax) : std::pair<T,T> (aMin,aMax) { }

  EgammaRange(const std::pair<T,T> & aPair ) : std::pair<T,T> (aPair) { }  

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

template <class T> std::ostream & operator<<( std::ostream& out, const egammaisolation::EgammaRange<T>& r) 
{ return out << "("<<r.min()<<","<<r.max()<<")"; }

#endif
