#ifndef CandUtils_Booster_h
#define CandUtils_Booster_h
/** \class Booster
 *
 * Boost a reco::Candidate by a specified boost vector 
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.8 $
 *
 * $Id: Booster.h,v 1.8 2006/03/03 10:09:18 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"

struct Booster : public reco::Candidate::setup {
  /// spatial vector
  typedef reco::Candidate::Vector Vector;
  /// constructor from a boost vector
  Booster( const Vector & b ) : 
    reco::Candidate::setup( setupCharge( false ), setupP4( true ), setupVertex( false ) ), 
    boost( b ) { }
  /// destructor
  virtual ~Booster();
  /// set up a candidate kinematics according to the boost
  virtual void set( reco::Candidate& c );
  /// the boost vector
  const Vector & boostVector() { return boost; }
private:
  const Vector boost;
};

#endif
