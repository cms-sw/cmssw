#ifndef Candidate_component_h
#define Candidate_component_h
/** \class reco::component
 *
 * Generic accessor to components of a Candidate 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: component.h,v 1.3 2006/04/27 05:59:58 llista Exp $
 *
 */
#include <boost/static_assert.hpp>
#include "FWCore/Utilities/interface/Exception.h"

namespace reco {

  class Candidate;

  struct DefaultComponentTag { };
  
  namespace componenthelper {

    struct VoidComponent {
      static size_t numberOf( const Candidate & c ) { return 0; }
    };

    template<typename C, typename T, T (C::*F)() const>
    struct SingleComponent {
      static size_t numberOf( const Candidate & c ) { return 1; }
      static T get( const Candidate & c ) {
	const C * dc = dynamic_cast<const C *>( & c );
	if ( dc == 0 ) return T();
	return (dc->*F)();
      }
    };

    template<typename C, typename T, T (C::*F)( size_t ) const , size_t (C::*S)() const>
    struct MultipleComponents {
      static size_t numberOf( const Candidate & c ) { 
	const C * dc = dynamic_cast<const C *>( & c );
	if ( dc == 0 ) return 0;
	return (dc->*S)(); 
      }
      static T get( const Candidate & c, size_t i ) {
	const C * dc = dynamic_cast<const C *>( & c );
	if ( dc == 0 ) return T();
	if ( i < (dc->*S)() ) return (dc->*F)( i );
	else throw cms::Exception( "Error" ) << "index " << i << " out ot range";
      }
    };

  }

  template<typename T, typename Tag = DefaultComponentTag>
  struct component {
    /// fail non specialized instances
    BOOST_STATIC_ASSERT(false);
  };

  template<typename T>
  inline T get( const Candidate & c ) {
    return component<T>::type::get( c );
  }

  template<typename T, typename Tag>
  inline T get( const Candidate & c ) {
    return component<T, Tag>::type::get( c );
  }

  template<typename T>
  inline T get( const Candidate & c, size_t i ) {
    return component<T>::type::get( c, i );
  }

  template<typename T, typename Tag>
  inline T get( const Candidate & c, size_t i ) {
    return component<T, Tag>::type::get( c, i );
  }

  template<typename T>
  inline size_t numberOf( const Candidate & c ) {
    return component<T>::type::numberOf( c );
  }

  template<typename T, typename Tag>
  inline size_t numberOf( const Candidate & c ) {
    return component<T, Tag>::type::numberOf( c );
  }

}

#define GET_CANDIDATE_COMPONENT( CAND, TYPE, TAG, FUN ) \
  template<> \
  struct  component<TYPE, TAG> { \
    typedef componenthelper::SingleComponent<CAND, TYPE, & CAND::FUN> type; \
  };

#define GET_CANDIDATE_MULTIPLECOMPONENTS( CAND, TYPE, TAG, FUN, SIZE ) \
  template<> \
  struct  component<TYPE, TAG> { \
    typedef componenthelper::MultipleComponents<CAND, TYPE, & CAND::FUN, & CAND::SIZE> type; \
  };

#endif
