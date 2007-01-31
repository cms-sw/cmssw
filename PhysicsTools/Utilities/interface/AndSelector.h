#ifndef RecoAlgos_AndSelector_h
#define RecoAlgos_AndSelector_h
/* \class OrSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AndSelector.h,v 1.1 2006/09/20 15:49:36 llista Exp $
 */
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

template<typename S1, typename S2>
struct AndSelector {
  BOOST_STATIC_ASSERT( ( boost::is_same<typename S1::value_type, typename S2::value_type>::value ) );
  typedef typename S1::value_type value_type;
  AndSelector( const S1 & s1, const S2 & s2 ) :
    s1_( s1 ), s2_( s2 ) { }
  bool operator()( const value_type & t ) const { 
    return s1_( t ) && s2_( t );
  }
private:
  S1 s1_;
  S2 s2_;
};

#endif
