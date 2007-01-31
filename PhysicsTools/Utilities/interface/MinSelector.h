#ifndef RecoAlgos_MinSelector_h
#define RecoAlgos_MinSelector_h
/* \class MinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MinSelector.h,v 1.2 2006/10/03 11:44:47 llista Exp $
 */

template<typename T, double (T::*fun)() const>
struct MinSelector {
  typedef T value_type;
  MinSelector( double min ) : 
    min_( min ) { }
  bool operator()( const value_type & t ) const { return (t.*fun)() >= min_; }
private:
  double min_;
};

#endif
