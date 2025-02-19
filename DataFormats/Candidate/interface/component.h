#ifndef Candidate_component_h
#define Candidate_component_h
/** \class reco::component
 *
 * Generic accessor to components of a Candidate 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: component.h,v 1.7 2006/12/11 10:12:01 llista Exp $
 *
 */
#include "FWCore/Utilities/interface/Exception.h"

namespace reco {

  class Candidate;

  struct DefaultComponentTag { };
  
  namespace componenthelper {

    struct SingleComponentTag { };

    struct MultipleComponentsTag { };

    template<typename C, typename T, T (C::*F)() const>
    struct SingleComponent {
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

  template<typename T, typename M, typename Tag = DefaultComponentTag>
  struct component { };

  template<typename T>
  inline T get( const Candidate & c ) {
    return component<T, componenthelper::SingleComponentTag>::type::get( c );
  }

  template<typename T, typename Tag>
  inline T get( const Candidate & c ) {
    return component<T, componenthelper::SingleComponentTag, Tag>::type::get( c );
  }

  template<typename T>
  inline T get( const Candidate & c, size_t i ) {
    return component<T, componenthelper::MultipleComponentsTag>::type::get( c, i );
  }

  template<typename T, typename Tag>
  inline T get( const Candidate & c, size_t i ) {
    return component<T, componenthelper::MultipleComponentsTag, Tag>::type::get( c, i );
  }

  template<typename T>
  inline size_t numberOf( const Candidate & c ) {
    return component<T, componenthelper::MultipleComponentsTag>::type::numberOf( c );
  }

  template<typename T, typename Tag>
  inline size_t numberOf( const Candidate & c ) {
    return component<T, componenthelper::MultipleComponentsTag, Tag>::type::numberOf( c );
  }

}

#define GET_CANDIDATE_COMPONENT( CAND, TYPE, FUN, TAG ) \
  template<> \
  struct  component<TYPE, componenthelper::SingleComponentTag, TAG> { \
    typedef componenthelper::SingleComponent<CAND, TYPE, & CAND::FUN> type; \
  }

#define GET_DEFAULT_CANDIDATE_COMPONENT( CAND, TYPE, FUN ) \
  template<> \
  struct  component<TYPE, componenthelper::SingleComponentTag, DefaultComponentTag> { \
    typedef componenthelper::SingleComponent<CAND, TYPE, & CAND::FUN> type; \
  }

#define GET_CANDIDATE_MULTIPLECOMPONENTS( CAND, TYPE, FUN, SIZE, TAG ) \
  template<> \
  struct  component<TYPE, componenthelper::MultipleComponentsTag, TAG> { \
    typedef componenthelper::MultipleComponents<CAND, TYPE, & CAND::FUN, & CAND::SIZE> type; \
  }

#define GET_DEFAULT_CANDIDATE_MULTIPLECOMPONENTS( CAND, TYPE, FUN, SIZE ) \
  template<> \
  struct  component<TYPE, componenthelper::MultipleComponentsTag, DefaultComponentTag> { \
    typedef componenthelper::MultipleComponents<CAND, TYPE, & CAND::FUN, & CAND::SIZE> type; \
  }

#endif
