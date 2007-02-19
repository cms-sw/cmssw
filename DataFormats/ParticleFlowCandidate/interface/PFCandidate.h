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

    /* ALEPH types : */
    /* 0 = hadrons chargés */
    /* 1 = electrons */
    /* 2 = muons */
    /* 3 = traces provenant de V0 */
    /* 4 = photons */
    /* 5 = hadrons neutres */
    /* 6 = FW detector particles  */

    /// particle types
    enum ParticleType {
      X=0,
      h,
      e,
      mu,
      gamma, 
      h0
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
