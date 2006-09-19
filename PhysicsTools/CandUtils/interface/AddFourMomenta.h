#ifndef CandUtils_AddFourMomenta_h
#define CandUtils_AddFourMomenta_h
/** \class AddFourMomenta
 *
 * set up a composite reco::Candidate adding its 
 * daughters four-momenta and electric charge
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.7 $
 *
 * $Id: AddFourMomenta.h,v 1.7 2006/07/26 08:48:05 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"

struct AddFourMomenta {
  /// set up a candidate
  void set( reco::Candidate& c ) const;
};

#endif
