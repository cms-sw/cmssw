#ifndef CommonTools_Utils_PointerComparator_h
#define CommonTools_Utils_PointerComparator_h
/** \class PoinetComparator
 *
 * adapt a comparator to take pointers as arguments
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: PointerComparator.h,v 1.1 2009/02/24 14:52:46 llista Exp $
 *
 */
#include "FWCore/Utilities/interface/EDMException.h"

template<typename C>
struct PointerComparator {
  typedef typename C::first_argument_type first_argument_type;
  typedef typename C::second_argument_type second_argument_type;
    bool operator()( const first_argument_type * t1, const second_argument_type * t2 ) const {
      if ( t1 == 0 || t2 == 0 )
	throw edm::Exception( edm::errors::NullPointerError )
	  << "PointerComparator: passed null pointer."; 
	return cmp( *t1, *t2 );
    }
  C cmp;
};


#endif
