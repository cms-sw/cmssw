#ifndef CandUtils_Booster_h
#define CandUtils_Booster_h
/** \class Booster
 *
 * Boost a reco::Candidate by a specified boost vector 
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.11 $
 *
 * $Id: Booster.h,v 1.11 2006/12/14 14:19:11 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/Vector3D.h"

struct Booster {
  /// constructor from a boost vector
  Booster( const math::XYZVector & b ) : boost( b ) { }
  /// set up a candidate kinematics according to the boost
  void set( reco::Candidate& c );
private:
  const math::XYZVector boost;
};

#endif
