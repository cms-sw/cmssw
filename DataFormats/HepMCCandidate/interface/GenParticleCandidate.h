#ifndef HepMCCandidate_GenParticleCandidate_h
#define HepMCCandidate_GenParticleCandidate_h
/** \class reco::GenParticleCandidate
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidate.h,v 1.4 2006/11/02 10:23:56 llista Exp $
 */
#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"

namespace HepMC {
  class GenParticle;
}

namespace reco {

  class GenParticleCandidate : public CompositeRefBaseCandidate {
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
    /// get candidate mother
    const GenParticleCandidateRef & mother() const { return mother_; }
    /// set mother reference
    void setMother( const GenParticleCandidateRef & ref ) const { mother_ = ref; }

  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
    /// PDG code
    int pdgId_;
    /// status code
    int status_;
    /// reference to mother
    mutable GenParticleCandidateRef mother_;
 };

}

#endif
