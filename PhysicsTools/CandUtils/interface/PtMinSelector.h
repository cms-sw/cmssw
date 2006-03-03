#ifndef CandUtils_PtMinSelector_h
#define CandUtils_PtMinSelector_h
/** \class PtMinSelector
 *
 * Selector that selects only candidates with a 
 * minumum value of the transverse momentum
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision$
 *
 * $Id: Track.h,v 1.12 2006/03/01 12:23:40 llista Exp $
 *
 */
#include "PhysicsTools/CandUtils/interface/CandSelector.h"

class PtMinSelector : public CandSelector {
public:
  /// constructor from minumum pt calue
  explicit PtMinSelector( double cut ) :
    ptMin( cut ) {
  }
  /// returns true if a candidate is selected
  bool operator()( const reco::Candidate & c ) const;
private:
  double ptMin;
};

#endif
