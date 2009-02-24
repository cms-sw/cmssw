#ifndef RecoAlgos_MaxSelector_h
#define RecoAlgos_MaxSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxSelector.h,v 1.5 2007/06/18 18:33:53 llista Exp $
 */

template<typename T>
struct MaxSelector {
  MaxSelector( T max ) : max_( max ) { }
  bool operator()( T t ) const { return t <= max_; }

private:
  T max_;
};

#endif
