#ifndef CandUtils_CenterOfMassBooster_h
#define CandUtils_CenterOfMassBooster_h
/** \class CenterOfMassBooster
 *
 * Boost a reco::Candidate to its center-of-mass reference frame
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision$
 *
 * $Id: Track.h,v 1.12 2006/03/01 12:23:40 llista Exp $
 *
 */
#include "PhysicsTools/CandUtils/interface/Booster.h"

struct CenterOfMassBooster : public Booster {
  /// constructor from a candidate
  CenterOfMassBooster( const reco::Candidate & c ) : Booster( c.boostToCM() ) {
  }
  /// destructor
  ~CenterOfMassBooster();
};

#endif
