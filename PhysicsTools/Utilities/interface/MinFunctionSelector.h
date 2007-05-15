#ifndef RecoAlgos_MinFunctionSelector_h
#define RecoAlgos_MinFunctionSelector_h
/* \class MinFunctionSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinSelector.h,v 1.3 2007/01/31 14:42:59 llista Exp $
 */

template<typename T, double (T::*fun)() const>
struct MinFunctionSelector {
  typedef T value_type;
  MinFunctionSelector( double min ) : 
    min_( min ) { }
  bool operator()( const value_type & t ) const { return (t.*fun)() >= min_; }
private:
  double min_;
};

#endif
