#ifndef CandUtils_ApplyEnergyCorrection_h
#define CandUtils_ApplyEnergyCorrection_h
/** \class ApplyEnergyCorrection
 *
 * apply correction factor to candidate energy 
 * and momenta, presenrving direction
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.6 $
 *
 * $Id: ApplyEnergyCorrection.h,v 1.6 2006/06/21 09:36:47 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"

struct ApplyEnergyCorrection : public reco::Candidate::setup {
  /// default constructor
  ApplyEnergyCorrection( double correction ) : 
    reco::Candidate::setup( setupCharge( false ), setupP4( true ), setupVertex( false ) ),
    correction_( correction ) { }
  /// destructor
  virtual ~ApplyEnergyCorrection();
  /// set up a candidate
  void set( reco::Candidate& c );
private:
  double correction_;
};

#endif
