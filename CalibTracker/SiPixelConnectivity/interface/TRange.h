#ifndef TRange_H
#define TRange_H

/** Define a range [aMin,aMax] */

#include <iostream>
#include <utility>
#include <algorithm>


template<class T> class TRange : public std::pair<T,T> {
public:

  TRange() { }

  TRange(const T & aMin, const T & aMax) 
      : std::pair<T,T> (aMin,aMax) { }

  TRange(const std::pair<T,T> & apair ) 
      : std::pair<T,T> (apair) { }  

  /// lower edge of range 
  const T & min() const { return this->first; }

  /// upper edge of range
  const T & max() const { return this->second; }

  T mean() const { return (this->first+this->second)/2.; }

  /// true for empty region. note that region [x,x] is not empty
  bool empty() const { return (this->second < this->first); }

  /// check if object is inside region
  bool inside(const T & value) const {
    if (value < this->first || this->second < value)  return false; 
    else  return true;
  }

/*
  /// check if ranges overlap
  bool hasIntersection( const TRange<T> & r) const {
    return rangesIntersect(*this,r); 
  }

  /// overlaping region
  TRange<T> intersection( 
      const TRange<T> & r) const {
    return rangeIntersection(*this,r); 
  }
*/

  /// sum of overlaping regions. the overlapping should be checked before.
  TRange<T> sum(const TRange<T> & r) const {
   if( this->empty()) return r;
   else if( r.empty()) return *this;
   else return TRange( (min() < r.min()) ? min() : r.min(), 
      (max() < r.max()) ? r.max() : max()); 
  }

  void sort() { if (empty() ) std::swap(this->first,this->second); }
};

template <class T> std::ostream & operator<<( 
    std::ostream& out, const TRange<T>& r) 
{
  return out << "("<<r.min()<<","<<r.max()<<")";
}
#endif
