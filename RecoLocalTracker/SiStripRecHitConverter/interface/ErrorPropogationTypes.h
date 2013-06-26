#ifndef ErrorPropogationTypes_h
#define ErrorPropogationTypes_h

#include "boost/operators.hpp"
#include <cmath>              

class count_t :
boost::addable<count_t,    
boost::incrementable<count_t, 
boost::totally_ordered<count_t, 
boost::equivalent<count_t> > > > {
  
 private: 
  unsigned long count;
 public:               
  count_t() : count(0) {}
  count_t(unsigned long n) : count(n) {}
  bool operator<(const count_t& R) const { return count < R.count;}
  count_t operator++() {++count; return *this;}                    
  count_t operator+=(const count_t& R) {count += R.count; return *this;}
  unsigned long operator()() const {return count;}                 
  unsigned long error2() const {return count;}                     
  double error() const {return sqrt(count);}                       
  double relative_error() const {return 1/sqrt(count);}            
};                                          
  
template <class charT, class traits> 
  inline
  std::basic_ostream<charT,traits>& operator<<(std::basic_ostream<charT,traits>& strm, const count_t& f)
{ strm << f() << "("<< f.error()<< ")"; return strm;}

template<class T>
class stats_t :  
    boost::arithmetic1<stats_t<T>, 
    boost::arithmetic2<stats_t<T>, T,
    boost::partially_ordered<stats_t<T>,
    boost::partially_ordered<stats_t<T>,count_t,
    boost::equality_comparable<stats_t<T>,      
    boost::equality_comparable<stats_t<T>,count_t> > > > > > {
  private:
  T value, err2;
  public:
  stats_t() : value(0), err2(1) {}
  stats_t(count_t c) : value(c()), err2(c.error2()) {}
  stats_t(T q, T e2) : value(q), err2(e2) {}
  static stats_t from_relative_uncertainty2(T q, T re2) { return stats_t(q,q*q*re2);}
  
  bool operator<(const stats_t& R) const { return value < R.value ;}
  bool operator<(const count_t& R) const { return value < R()     ;}
  bool operator==(const stats_t& R) const { return value == R.value && err2 == R.err2;}
  bool operator==(const count_t& R) const { return value == R()     && err2 == R.error2();}
  
  stats_t operator+=(const stats_t& R) { value += R.value;   err2 += R.err2; return *this;}
  stats_t operator-=(const stats_t& R) { value -= R.value;   err2 += R.err2; return *this;}
  stats_t operator*=(const stats_t& R) { err2 = R.err2*value*value + err2*R.value*R.value;        value *= R.value;  return *this;}
  stats_t operator/=(const stats_t& R) { value /= R.value;  err2 = (err2 + value*value*R.err2) / (R.value*R.value);  return *this;}
  
  stats_t operator+=(const T& r) { value += r; return *this;}
  stats_t operator-=(const T& r) { value -= r; return *this;}
  stats_t operator*=(const T& r) { value *= r; err2 *= r*r; return *this;}
  stats_t operator/=(const T& r) { value /= r; err2 /= r*r; return *this;}
  
  stats_t inverse() const { return stats_t(1./value, err2/pow(value,4));}
  
  T operator()() const {return value;}
  T error2() const {return err2;}
  T error() const {return sqrt(err2);}
  T relative_error() const {return sqrt(err2)/value;}
  T sigmaFrom(const T& x) const {return fabs(value-x)/error();}
};

template <class charT, class traits, class T> inline
  std::basic_ostream<charT,traits>& operator<<(std::basic_ostream<charT,traits>& strm, const stats_t<T>& f)
{ strm << f() << "("<< f.error()<< ")"; return strm; }

#endif
