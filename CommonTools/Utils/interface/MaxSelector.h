#ifndef RecoAlgos_MaxSelector_h
#define RecoAlgos_MaxSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

template<typename T>
struct MaxSelector {
  MaxSelector( T max ) : max_( max ) { }
  bool operator()( T t ) const { return t <= max_; }

private:
  T max_;
};

#endif
