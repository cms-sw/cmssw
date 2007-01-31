#ifndef RecoAlgos_EtMinSelector_h
#define RecoAlgos_EtMinSelector_h
/* \class EtMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: EtMinSelector.h,v 1.3 2006/10/03 12:06:23 llista Exp $
 */

template<typename T>
struct EtMinSelector {
  typedef T value_type;
  EtMinSelector( double etMin ) : 
    etMin_( etMin ) { }
  bool operator()( const value_type & t ) const { return t.et() >= etMin_; }
private:
  double etMin_;
};

#endif
