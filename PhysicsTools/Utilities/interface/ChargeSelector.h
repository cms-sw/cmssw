#ifndef RecoAlgos_ChargeSelector_h
#define RecoAlgos_ChargeSelector_h
/* \class ChargeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: ChargeSelector.h,v 1.2 2007/01/31 14:42:59 llista Exp $
 */

struct ChargeSelector {
  ChargeSelector( int charge ) : charge_( charge ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return ( t.charge() == charge_ ); 
  }

private:
  int charge_;
};

#endif
