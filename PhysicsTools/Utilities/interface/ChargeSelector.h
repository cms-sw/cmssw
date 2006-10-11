#ifndef RecoAlgos_ChargeSelector_h
#define RecoAlgos_ChargeSelector_h
/* \class ChargeSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: ChargeSelector.h,v 1.2 2006/10/03 10:34:03 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename T>
struct ChargeSelector {
  typedef T value_type;
  ChargeSelector( int charge ) : 
    charge_( charge ) { }
  explicit ChargeSelector( const edm::ParameterSet & cfg ) : 
    charge_( cfg.template getParameter<int>( "charge" ) ) {
  }
  bool operator()( const value_type & t ) const { 
    return ( t.charge() == charge_ ); 
  }
private:
  int charge_;
};

#endif
