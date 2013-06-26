#ifndef CandUtils_ApplyEnergyCorrection_h
#define CandUtils_ApplyEnergyCorrection_h
/** \class ApplyEnergyCorrection
 *
 * apply correction factor to candidate energy 
 * and momenta, presenrving direction
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ApplyEnergyCorrection.h,v 1.1 2009/02/26 09:17:33 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"

struct ApplyEnergyCorrection {
  ApplyEnergyCorrection( double correction ) : correction_( correction ) { }
  /// set up a candidate
  void set( reco::Candidate& c );
  
private:
  double correction_;
};

#endif
