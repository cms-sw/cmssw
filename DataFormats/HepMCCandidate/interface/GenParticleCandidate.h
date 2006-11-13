#ifndef HepMCCandidate_GenParticleCandidate_h
#define HepMCCandidate_GenParticleCandidate_h
/** \class reco::GenParticleCandidate
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidate.h,v 1.6 2006/11/03 18:30:10 llista Exp $
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
    /// get candidate mother
    CandidateRef mother() const { return mother_; }
    /// set mother reference
    void setMother( const CandidateRef & ref ) const { mother_ = ref; }

  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
    /// PDG code
    int pdgId_;
    /// status code
    int status_;
    /// reference to mother
    mutable CandidateRef mother_;
 };

  /// PDG id component tag
  struct PdgIdTag { };

  /// status code component tag
  struct StatusTag { };

  /// mother reference component tag
  struct MotherRefTag { };

  /// get PDG id component
  GET_CANDIDATE_COMPONENT( GenParticleCandidate, int, pdgId, PdgIdTag );

  /// get status code component
  GET_CANDIDATE_COMPONENT( GenParticleCandidate, int, status, StatusTag );

  /// get mother reference component
  GET_CANDIDATE_COMPONENT( GenParticleCandidate, CandidateRef, mother, MotherRefTag );

  inline int pdgId( const Candidate & c ) {
    return c.get<int, PdgIdTag>();
  }

  inline int status( const Candidate & c ) {
    return c.get<int, StatusTag>();
  }

  inline CandidateRef mother( const Candidate & c ) {
    return c.get<CandidateRef, MotherRefTag>();
  }

}

#endif
