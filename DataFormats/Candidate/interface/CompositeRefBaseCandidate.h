#ifndef Candidate_CompositeRefBaseCandidate_h
#define Candidate_CompositeRefBaseCandidate_h
#include "DataFormats/Candidate/interface/LeafCandidate.h"
/** \class reco::CompositeRefBaseCandidate
 *
 * a reco::Candidate composed of daughters. 
 * The daughters has persistent references (edm::RefToBase<...>) 
 * to reco::Candidate stored in a separate collection.
 *
 * \author Luca Lista, INFN
 *
 *
 */

namespace reco {

  class CompositeRefBaseCandidate : public LeafCandidate {
  public:
    /// collection of references to daughters
    typedef std::vector<CandidateBaseRef> daughters;
    /// default constructor
    CompositeRefBaseCandidate() : LeafCandidate() { }
    /// constructor from values
    CompositeRefBaseCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			       int pdgId = 0, int status = 0, bool integerCharge = true ) :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from values
    CompositeRefBaseCandidate( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			       int pdgId = 0, int status = 0, bool integerCharge = true ) :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from a particle
    explicit CompositeRefBaseCandidate( const Candidate & c ) : LeafCandidate( c ) { }
    /// destructor
    ~CompositeRefBaseCandidate() override;
    /// returns a clone of the candidate
    CompositeRefBaseCandidate * clone() const override;
    /// number of daughters
    size_t numberOfDaughters() const override;
    /// number of mothers
    size_t numberOfMothers() const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    const Candidate * daughter( size_type ) const override;
    /// return mother at a given position, i = 0, ... numberOfMothers() - 1 (read only mode)
    const Candidate * mother( size_type ) const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    Candidate * daughter( size_type ) override;
    using reco::LeafCandidate::daughter; // avoid hiding the base
    /// add a daughter via a reference
    void addDaughter( const CandidateBaseRef & );    
    /// clear daughter references
    void clearDaughters() { dau.clear(); }
    /// reference to daughter at given position
    CandidateBaseRef daughterRef( size_type i ) const { return dau[ i ]; }

  private:
    /// collection of references to daughters
    daughters dau;
    /// check overlap with another candidate
    bool overlap( const Candidate & ) const override;
  };

  inline void CompositeRefBaseCandidate::addDaughter( const CandidateBaseRef & cand ) { 
    dau.push_back( cand ); 
  }
}

#endif
