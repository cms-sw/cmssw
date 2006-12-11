#ifndef HepMCCandidate_GenParticleCandidate_h
#define HepMCCandidate_GenParticleCandidate_h
/** \class reco::GenParticleCandidate
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidate.h,v 1.10 2006/12/07 18:35:49 llista Exp $
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
    CandidateRef motherRef() const { return mother_; }
    /// set mother reference
    void setMotherRef( const CandidateRef & ref ) { mother_ = ref; }
    /// mother reference
    virtual const Candidate * mother() const; 	 
  private:
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
    /// PDG code
    int pdgId_;
    /// status code
    int status_;
    /// reference to mother
    CandidateRef mother_;
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

  inline int pdgId( const Candidate & c ) {
    return c.get<int, PdgIdTag>();
  }

  inline int status( const Candidate & c ) {
    return c.get<int, StatusTag>();
  }

}

#endif
