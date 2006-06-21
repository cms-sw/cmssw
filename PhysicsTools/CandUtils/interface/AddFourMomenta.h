#ifndef CandUtils_AddFourMomenta_h
#define CandUtils_AddFourMomenta_h
/** \class AddFourMomenta
 *
 * set up a composite reco::Candidate adding its 
 * daughters four-momenta and electric charge
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.5 $
 *
 * $Id: AddFourMomenta.h,v 1.5 2006/03/03 10:09:18 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"

struct AddFourMomenta : public reco::Candidate::setup {
  /// default constructor
  AddFourMomenta() : reco::Candidate::setup( setupCharge( true ), setupP4( true ), setupVertex( false ) ) { }
  /// destructor
  virtual ~AddFourMomenta();
  /// set up a candidate
  void set( reco::Candidate& c );
};

#endif
