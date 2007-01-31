#ifndef RecoAlgos_MassRangeSelector_h
#define RecoAlgos_MassRangeSelector_h
/* \class MassRangeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: MassRangeSelector.h,v 1.1 2006/10/11 08:50:36 llista Exp $
 */

template<typename T>
struct MassRangeSelector {
  typedef T value_type;
  MassRangeSelector( double massMin, double massMax ) : 
    massMin_( massMin ), massMax_( massMax ) { }
  bool operator()( const value_type & t ) const { 
    double mass = t.mass();
    return ( mass >= massMin_ && mass <= massMax_ ); 
  }

private:
  double massMin_, massMax_;
};

#endif
