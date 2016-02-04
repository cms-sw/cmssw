#ifndef SPECIAL_LESS
#define SPECIAL_LESS

#include <functional>

struct fabs_less { 
  bool operator()(const double x, const double y) const { 
    return fabs(x) < fabs(y); 
  } 
};

template <class T> 
struct pt_less : std::binary_function <T,T,bool> {
  bool operator() (const T& x, const T& y) const
  {return x.Pt() < y.Pt();}
};

template <class T> 
struct pair2_less : std::binary_function <T,T,bool> {
  bool operator() (const T& x, const T& y) const
  {return x.second < y.second;}
};

#endif
