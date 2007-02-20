#ifndef ParticleFlowCandidate_PFCandidate_h
#define ParticleFlowCandidate_PFCandidate_h
/** \class reco::PFCandidate
 *
 * particle candidate from particle flow
 *
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace reco {

  class PFCandidate : public LeafCandidate {
  public:

    /// particle types
    enum ParticleType {
      X=0,     // undefined
      h,       // charged hadron
      e,       // electron 
      mu,      // muon 
      gamma,   // photon
      h0       // neutral hadron
    };

    /// default constructor
    PFCandidate() : LeafCandidate(), particleId_( X ) { }

    PFCandidate(Charge q, 
		const LorentzVector & p4, 
		ParticleType particleId ) : 
      LeafCandidate(q, p4), 
      particleId_(particleId) { }

    /// destructor
    virtual ~PFCandidate() {}

    /// return a clone
    PFCandidate * clone() const;

    /// particle identification
    virtual int particleId() const { return particleId_;}

  private:

    /// particle identification
    ParticleType particleId_; 
  };

}

#endif
