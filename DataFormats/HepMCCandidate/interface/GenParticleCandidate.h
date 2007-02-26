#ifndef HepMCCandidate_GenParticleCandidate_h
#define HepMCCandidate_GenParticleCandidate_h
/** \class reco::GenParticleCandidate
 *
 * particle candidate with information from HepMC::GenParticle
 *
 * \author: Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidate.h,v 1.15 2007/02/19 12:59:05 llista Exp $
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
    /// constrocturo from values
    GenParticleCandidate( Charge q, const LorentzVector & p4, const Point & vtx, 
			  int pdgId, int status, bool integerCharge = true );
    /// destructor
    virtual ~GenParticleCandidate();
    /// return a clone
    GenParticleCandidate * clone() const;
    /// PDG code
    virtual int pdgId() const { return pdgId_; }
    /// status code
    int status() const { return status_; }

  private:
    /// post-read fixup
    virtual void fixup() const;
    /// checp overlap with another candidate
    bool overlap( const Candidate & ) const;
    /// PDG code
    int pdgId_;
    /// status code
    int status_;
 };

  /// PDG id component tag
  struct PdgIdTag { };

  /// status code component tag
  struct StatusTag { };

  /// get PDG id component
  GET_CANDIDATE_COMPONENT( GenParticleCandidate, int, pdgId, PdgIdTag );

  /// get status code component
  GET_CANDIDATE_COMPONENT( GenParticleCandidate, int, status, StatusTag );

  /// left for backward compatibility. Can now be replaced by
  /// an equivalent member function
  inline int pdgId( const Candidate & c ) {
    return c.pdgId();
  }

  inline int status( const Candidate & c ) {
    return c.get<int, StatusTag>();
  }

}

#endif
