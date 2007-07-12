#ifndef RecoAlgos_MassMinSelector_h
#define RecoAlgos_MassMinSelector_h
/* \class MassMinSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MassMinSelector.h,v 1.3 2007/06/18 18:33:53 llista Exp $
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
