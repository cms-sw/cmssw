#ifndef RecoAlgos_MassMinSelector_h
#define RecoAlgos_MassMinSelector_h
/* \class MassMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MassMinSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
 */

struct MassMinSelector {
  MassMinSelector( double massMin ) : 
    massMin_( massMin ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return t.mass() >= massMin_; 
  }

private:
  double massMin_;
};

#endif
