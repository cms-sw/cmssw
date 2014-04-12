#ifndef CommonTools_CandUtils_AddFourMomenta_h
#define CommonTools_CandUtils_AddFourMomenta_h
/** \class AddFourMomenta
 *
 * set up a composite reco::Candidate adding its 
 * daughters four-momenta and electric charge
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.9 $
 *
 * $Id: AddFourMomenta.h,v 1.9 2007/01/09 09:14:42 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

struct AddFourMomenta {
  /// default constructor
  AddFourMomenta() { }
  /// constructor
  explicit AddFourMomenta( const edm::ParameterSet & ) { }
  /// set up a candidate
  void set( reco::Candidate& c ) const;
};

#endif
