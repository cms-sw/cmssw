#ifndef RecoAlgos_MaxSelector_h
#define RecoAlgos_MaxSelector_h
/* \class MaxSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MaxSelector.h,v 1.2 2012/06/26 21:13:12 wmtan Exp $
 */

template<typename T>
struct MaxSelector {
  MaxSelector( T max ) : max_( max ) { }
  bool operator()( T t ) const { return t <= max_; }

private:
  T max_;
};

#endif
