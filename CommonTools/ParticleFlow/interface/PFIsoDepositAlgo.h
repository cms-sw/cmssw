#ifndef CommonTools_ParticleFlow_PFIsoDepositAlgo_
#define CommonTools_ParticleFlow_PFIsoDepositAlgo_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//not a fwd declaration, to save the pain to the user to include the necessary DF header as well
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

/**\class PFIsoDepositAlgo 
\brief Computes the iso deposits for a collection of PFCandidates. 

\author Colin Bernet
\date   february 2008
*/

namespace pf2pat {
  
  class PFIsoDepositAlgo {
  public:
    
    // can be a template parameter (IsoDeposits from GenParticles? )
    typedef reco::PFCandidate   Particle;
    typedef std::vector< Particle > ParticleCollection;
    
    // random access to the IsoDeposit corresponding to a given particle
    typedef std::vector< reco::IsoDeposit > IsoDeposits;

    explicit PFIsoDepositAlgo(const edm::ParameterSet&);

    ~PFIsoDepositAlgo();
    
    /// all the filtering is done before
    /// could gain in performance by having a single loop, and 
    /// by producing all isodeposits at the same time?
    /// however, would not gain in ease of maintenance, and in flexibility.
    const IsoDeposits&  produce(const ParticleCollection& toBeIsolated,
				const ParticleCollection& forIsolation );
    
  private:    
    
    /// build the IsoDeposit for "particle"
    reco::IsoDeposit buildIsoDeposit( const Particle& particle, 
				      const ParticleCollection& forIsolation ) const; 
    
    /// checks if the 2 particles are in fact the same
    bool sameParticle( const Particle& particle1,
		       const Particle& particle2 ) const;

    
    /// IsoDeposits computed in the produce function
    IsoDeposits  isoDeposits_;

    bool verbose_;
  };
}

#endif
