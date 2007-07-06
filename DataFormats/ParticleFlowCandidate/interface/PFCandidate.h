#ifndef ParticleFlowCandidate_PFCandidate_h
#define ParticleFlowCandidate_PFCandidate_h
/** \class reco::PFCandidate
 *
 * particle candidate from particle flow
 *
 */

#include <iostream>

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"

namespace reco {
  /**\class PFCandidate
     \brief Particle reconstructed by the particle flow algorithm.
          
     \author Colin Bernet
     \date   February 2007
  */
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
    PFCandidate() : particleId_( X ) { }

    PFCandidate(Charge q, 
		const LorentzVector & p4, 
		ParticleType particleId, 
		reco::PFBlockRef blockref ) : 
      LeafCandidate(q, p4), 
      particleId_(particleId), 
      blockRef_(blockref) {}
    
    /// destructor
    virtual ~PFCandidate() {}

    /// return a clone
    virtual PFCandidate * clone() const;
    
    /// particle identification
    virtual int particleId() const { return particleId_;}
    

    /// return reference to the block
    const reco::PFBlockRef & blockRef() const { return blockRef_; } 

    /// return reference to the block
    PFBlockRef block() const { return blockRef_; } 
    
    friend std::ostream& operator<<( std::ostream& out, 
				     const PFCandidate& c );
    

  private:
    
    /// particle identification
    ParticleType particleId_; 
    
    /// reference to the corresponding PFBlock
    reco::PFBlockRef blockRef_;
  };

  /// particle ID component tag
  struct PFParticleIdTag { };

  /// get default PFBlockRef component
  /// as: pfcand->get<PFBlockRef>();
  GET_DEFAULT_CANDIDATE_COMPONENT( PFCandidate, PFBlockRef, block );

  /// get int component
  /// as: pfcand->get<int, PFParticleIdTag>();
  GET_CANDIDATE_COMPONENT( PFCandidate, int, particleId, PFParticleIdTag );

}

#endif
