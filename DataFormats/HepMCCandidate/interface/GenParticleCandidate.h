#ifndef HepMCCandidate_GenParticleCandidate_h
#define HepMCCandidate_GenParticleCandidate_h
/** \class reco::GenParticleCandidate
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: HepMCCandidate.h,v 1.6 2006/07/27 07:13:41 llista Exp $
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
    /// constroctor from pointer to generator particle
    GenParticleCandidate( const HepMC::GenParticle * );
    /// destructor
    virtual ~GenParticleCandidate();
    /// return a clone
    GenParticleCandidate * clone() const;
    /// PDG code
    int pdgId() const { return pdgId_; }
    /// status code
    int status() const { return status_; }
  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
    /// PDG code
    int pdgId_;
    /// status code
    int status_;
  };

}

#endif
