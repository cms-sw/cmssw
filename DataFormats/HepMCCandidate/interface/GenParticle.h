#ifndef HepMCCandidate_GenParticle_h
#define HepMCCandidate_GenParticle_h
/** \class reco::GenParticle
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: GenParticle.h,v 1.19 2007/05/14 11:47:17 llista Exp $
 */
#include "DataFormats/Candidate/interface/CompositeRefCandidateT.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include <vector>

namespace HepMC {
  class GenParticle;
}

namespace reco {

  class GenParticle : public CompositeRefCandidateT<GenParticleRefVector> {
  public:
    /// default constructor
    GenParticle() { }
    /// constrocturo from values
    GenParticle( Charge q, const LorentzVector & p4, const Point & vtx, 
			  int pdgId, int status, bool integerCharge );
    /// destructor
    virtual ~GenParticle();
    /// return a clone
    GenParticle * clone() const;

  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
 };

}

#endif
