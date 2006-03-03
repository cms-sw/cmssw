#ifndef CandUtils_MassWindowSelector_h
#define CandUtils_MassWindowSelector_h
/** \class MassWindowSelector
 *
 * Selector that selects only candidates with a 
 * mass within the specified window
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision$
 *
 * $Id: Track.h,v 1.12 2006/03/01 12:23:40 llista Exp $
 *
 */
#include "PhysicsTools/CandUtils/interface/CandSelector.h"

class MassWindowSelector : public CandSelector {
public:
  /// constructor from minumum and maximum mass values
  explicit MassWindowSelector( double massMin, double massMax ) :
    mMin2( massMin ), mMax2( massMax ) {
    mMin2 *= mMin2;
    mMax2 *= mMax2;
  }
  /// returns trye if a candidate is selected
  bool operator()( const reco::Candidate & c ) const;
private:
  double mMin2, mMax2;
};

#endif
