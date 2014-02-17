#ifndef RecoAlgos_EtMinSelector_h
#define RecoAlgos_EtMinSelector_h
/* \class EtMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: EtMinSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

struct EtMinSelector {
  EtMinSelector( double etMin ) : etMin_( etMin ) { }
  template<typename T>
  bool operator()( const T & t ) const { return t.et() >= etMin_; }

private:
  double etMin_;
};

#endif
