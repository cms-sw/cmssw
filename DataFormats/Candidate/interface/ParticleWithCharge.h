#ifndef Candidate_ParticleWithCharge_h
#define Candidate_ParticleWithCharge_h
/** \class reco::ParticleWithCharge
 *
 * Base class describing a generic reconstructed particle
 * with 4-momentum and charge measurements
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ParticleWithCharge.h,v 1.1 2006/06/05 13:40:07 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/ParticleKinematics.h"
#include "DataFormats/Math/interface/Point3D.h"
 
namespace reco {

  class ParticleWithCharge : public ParticleKinematics {
  public:
    /// electric charge type
    typedef char Charge;
    /// default constructor
    ParticleWithCharge() { }
    /// constructor from values
    ParticleWithCharge( Charge q, const LorentzVector & p4 ) : 
      ParticleKinematics( p4 ), q_( q ) { }
    /// destructor
    virtual ~ParticleWithCharge() { }
    /// electric charge
    int charge() const { return q_; }
  protected:
    /// electric charge
    Charge q_;    
  };

}

#endif
