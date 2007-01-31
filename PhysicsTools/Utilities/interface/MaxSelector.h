#ifndef RecoAlgos_MaxSelector_h
#define RecoAlgos_MaxSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxSelector.h,v 1.2 2006/10/03 12:06:23 llista Exp $
 */

template<typename T, double (T::*fun)() const>
struct MaxSelector {
  typedef T value_type;
  MaxSelector( double max ) : 
    max_( max ) { }
  bool operator()( const value_type & t ) const { return (t.*fun)() <= max_; }

private:
  double max_;
};

#endif
