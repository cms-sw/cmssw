#ifndef Candidate_component_h
#define Candidate_component_h
/** \class reco::component
 *
 * Generic accessor to components of a Candidate 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: component.h,v 1.1 2006/03/08 12:57:07 llista Exp $
 *
 */
#include <boost/static_assert.hpp>

namespace reco {

  struct DefaultComponentTag { 
  };
  
  template<typename T, typename Tag = DefaultComponentTag>
  struct component {
    /// fail non specialized instances
    BOOST_STATIC_ASSERT(false);
  };

}

#define GET_CANDIDATE_COMPONENT( CAND, TYPE, TAG, FUN ) \
  template<> \
  struct  component<TYPE, TAG> { \
    static TYPE get( const Candidate & c ) { \
      const CAND * dc = dynamic_cast<const CAND *>( & c ); \
      if ( dc == 0 ) return TYPE(); \
      return dc->FUN(); \
    } \
  };

#endif
