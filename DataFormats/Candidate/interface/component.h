#ifndef Candidate_Candidate_h
#define Candidate_Candidate_h
/** \class reco::component
 *
 * Generic accessor to components of a Candidate 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Candidate.h,v 1.2 2006/03/08 12:26:37 llista Exp $
 *
 */
#include <boost/static_assert.hpp>

namespace reco {
  
  template<typename T>
  struct component {
    /// fail non specialized instances
    BOOST_STATIC_ASSERT(false);
  };

}

#endif
