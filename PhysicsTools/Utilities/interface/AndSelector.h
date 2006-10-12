#ifndef RecoAlgos_AndSelector_h
#define RecoAlgos_AndSelector_h
/* \class OrSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: PtMinSelector.h,v 1.2 2006/07/25 17:21:31 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

template<typename S1, typename S2>
struct AndSelector {
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S2::value_type>::value ) );
  typedef typename S1::value_type value_type;
  AndSelector( const edm::ParameterSet & cfg ) : 
    s1_( cfg ), s2_( cfg ) { }
  bool operator()( const value_type & t ) const { 
    return s1_( t ) && s2_( t );
  }
private:
  S1 s1_;
  S2 s2_;
};

#endif
