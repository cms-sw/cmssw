#ifndef Candidate_ParticleBaseWithCharge_h
#define Candidate_ParticleBaseWithCharge_h
/** \class reco::ParticleBaseWithCharge
 *
 * Base class describing a generic reconstructed particle
 * with 4-momentum and charge measurements
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ParticleBaseWithCharge.h,v 1.3 2006/05/02 16:13:33 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/ParticleBase.h"
#include "DataFormats/Math/interface/Point3D.h"
 
namespace reco {

  class ParticleBaseWithCharge : public ParticleBase {
  public:
    /// electric charge type
    typedef char Charge;
    /// default constructor
    ParticleBaseWithCharge() { }
    /// constructor from values
    ParticleBaseWithCharge( Charge q, const LorentzVector & p4 ) : 
      ParticleBase( p4 ), q_( q ) { }
    /// destructor
    virtual ~ParticleBaseWithCharge() { }
    /// electric charge
    int charge() const { return q_; }
  protected:
    /// electric charge
    Charge q_;    
  };

}

#endif
