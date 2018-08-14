#ifndef Candidate_CompositeRefCandidate_h
#define Candidate_CompositeRefCandidate_h
#include "DataFormats/Candidate/interface/LeafCandidate.h"
/** \class reco::CompositeRefCandidate
 *
 * a reco::Candidate composed of daughters. 
 * The daughters has persistent references (edm::Ref <...>) 
 * to reco::Candidate stored in a separate collection.
 *
 * \author Luca Lista, INFN
 *
 *
 */

namespace reco {

  class CompositeRefCandidate : public LeafCandidate {
  public:
    /// collection of references to daughters
    typedef CandidateRefVector daughters;
    /// collection of references to daughters
    typedef CandidateRefVector mothers;
    /// default constructor
    CompositeRefCandidate() : LeafCandidate() { }
    /// constructor from values
    CompositeRefCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from values
    CompositeRefCandidate( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from a candidate
    explicit CompositeRefCandidate( const Candidate & p ) : LeafCandidate( p ) { }
    /// destructor
    ~CompositeRefCandidate() override;
    /// returns a clone of the candidate
    CompositeRefCandidate * clone() const override;
    /// number of daughters
    size_t numberOfDaughters() const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    const Candidate * daughter( size_type ) const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    Candidate * daughter( size_type ) override;
    using reco::LeafCandidate::daughter; // avoid hiding the base
    /// add a daughter via a reference
    void addDaughter( const CandidateRef & );    
    /// add a daughter via a reference
    void addMother( const CandidateRef & );    
    /// clear daughter references
    void clearDaughters() { dau.clear(); }
    /// reference to daughter at given position
    CandidateRef daughterRef( size_type i ) const { return dau[ i ]; }
    /// references to daughtes
    const daughters & daughterRefVector() const { return dau; }
    /// reference to mother at given position
    CandidateRef motherRef( size_type i = 0 ) const { return mom[ i ]; }
    /// references to mothers
    const mothers & motherRefVector() const { return mom; }
    /// set daughters product ID
    void resetDaughters( const edm::ProductID & id ) { dau = daughters( id ); }
    /// number of mothers (zero or one in most of but not all the cases)
    size_t numberOfMothers() const override;
    /// return pointer to mother
    const Candidate * mother( size_t i = 0 ) const override;

  private:
    /// collection of references to daughters
    daughters dau;
    /// collection of references to mothers
    daughters mom;
    /// check overlap with another candidate
    bool overlap( const Candidate & ) const override;
  };

  inline void CompositeRefCandidate::addDaughter( const CandidateRef & cand ) { 
    dau.push_back( cand ); 
  }

  inline void CompositeRefCandidate::addMother( const CandidateRef & cand ) { 
    mom.push_back( cand ); 
  }
}

#endif
