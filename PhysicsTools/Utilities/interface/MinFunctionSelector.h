#ifndef RecoAlgos_MinFunctionSelector_h
#define RecoAlgos_MinFunctionSelector_h
/* \class MinFunctionSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinFunctionSelector.h,v 1.1 2007/05/15 16:07:53 llista Exp $
 */

template<typename T, double (T::*fun)() const>
struct MinFunctionSelector {
  MinFunctionSelector( double min ) : 
    min_( min ) { }
  bool operator()( const T & t ) const { return (t.*fun)() >= min_; }
private:
  double min_;
};

#endif
