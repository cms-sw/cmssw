#ifndef HepMCCandidate_GenParticleCandidate_h
#define HepMCCandidate_GenParticleCandidate_h
/** \class reco::GenParticleCandidate
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidate.h,v 1.19 2007/05/14 11:47:17 llista Exp $
 */
#include "DataFormats/Candidate/interface/CompositeRefCandidate.h" 

namespace HepMC {
  class GenParticle;
}

namespace reco {

  class GenParticleCandidate : public CompositeRefCandidate {
  public:
    /// default constructor
    GenParticleCandidate() { }
    /// constrocturo from values
    GenParticleCandidate( Charge q, const LorentzVector & p4, const Point & vtx, 
			  int pdgId, int status, bool integerCharge );
    /// constrocturo from values
    GenParticleCandidate( Charge q, const PolarLorentzVector & p4, const Point & vtx, 
			  int pdgId, int status, bool integerCharge );
    /// destructor
    virtual ~GenParticleCandidate();
    /// return a clone
    GenParticleCandidate * clone() const;

  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
 };

}

#endif
