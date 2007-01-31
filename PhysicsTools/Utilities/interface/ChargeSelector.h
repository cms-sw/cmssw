#ifndef RecoAlgos_ChargeSelector_h
#define RecoAlgos_ChargeSelector_h
/* \class ChargeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: ChargeSelector.h,v 1.1 2006/10/11 08:50:35 llista Exp $
 */

template<typename T>
struct ChargeSelector {
  typedef T value_type;
  ChargeSelector( int charge ) : 
    charge_( charge ) { }
  bool operator()( const value_type & t ) const { 
    return ( t.charge() == charge_ ); 
  }
private:
  int charge_;
};

#endif
