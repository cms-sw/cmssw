#ifndef RecoAlgos_MaxSelector_h
#define RecoAlgos_MaxSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxSelector.h,v 1.3 2007/01/31 14:42:59 llista Exp $
 */

template<typename T>
struct MaxSelector {
  typedef T value_type;
  MaxSelector( T max ) : max_( max ) { }
  bool operator()( T t ) const { return t <= max_; }

private:
  T max_;
};

#endif
