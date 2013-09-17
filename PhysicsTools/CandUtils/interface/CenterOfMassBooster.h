#ifndef CandUtils_CenterOfMassBooster_h
#define CandUtils_CenterOfMassBooster_h
/** \class CenterOfMassBooster
 *
 * Boost a reco::Candidate to its center-of-mass reference frame
 *
 * \author Luca Lista, INFN
 *
 *
 *
 */
#include "PhysicsTools/CandUtils/interface/Booster.h"

struct CenterOfMassBooster {
  /// constructor from a candidate
  CenterOfMassBooster( const reco::Candidate & c );
  /// set up a candidate kinematics according to the boost
  void set( reco::Candidate& c ) { booster.set( c ); }
private:
  Booster booster;
};

#endif
