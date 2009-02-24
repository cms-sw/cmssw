#ifndef CommonTools_Utils_ChargeSelector_h
#define CommonTools_Utils_ChargeSelector_h
/* \class ChargeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: ChargeSelector.h,v 1.3 2007/06/18 18:33:53 llista Exp $
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
