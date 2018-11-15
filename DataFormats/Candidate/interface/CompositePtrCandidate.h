#ifndef Candidate_CompositePtrCandidate_h
#define Candidate_CompositePtrCandidate_h
#include "DataFormats/Candidate/interface/LeafCandidate.h"
/** \class reco::CompositePtrCandidate
 *
 * a reco::Candidate composed of daughters. 
 * The daughters has persistent references (edm::Ptr <...>) 
 * to reco::Candidate stored in a separate collection.
 *
 * \author Luca Lista, INFN
 *
 *
 */

namespace reco {

  class CompositePtrCandidate : public LeafCandidate {
  public:
    /// collection of references to daughters
    typedef std::vector<CandidatePtr> daughters;
    /// collection of references to daughters
    typedef std::vector<CandidatePtr> mothers;
    /// default constructor
    CompositePtrCandidate() : LeafCandidate() { }
    /// constructor from values
    CompositePtrCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from values
    CompositePtrCandidate( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      LeafCandidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from a Candidate
    explicit CompositePtrCandidate( const Candidate & p ) : LeafCandidate( p ) { }
    /// destructor
    ~CompositePtrCandidate() override;
    /// returns a clone of the candidate
    CompositePtrCandidate * clone() const override;
    /// number of daughters
    size_t numberOfDaughters() const override;
    /// number of mothers
    size_t numberOfMothers() const override;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    const Candidate * daughter( size_type ) const override;
    using reco::LeafCandidate::daughter; // avoid hiding the base
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    Candidate * daughter( size_type ) override;
    /// add a daughter via a reference
    void addDaughter( const CandidatePtr & );    
    /// clear daughter references
    virtual void clearDaughters() { dau.clear(); }
    /// reference to daughter at given position
    virtual CandidatePtr daughterPtr( size_type i ) const { return dau[ i ]; }
    /// references to daughtes
    virtual const daughters & daughterPtrVector() const { return dau; }
    /// return pointer to mother
    const Candidate * mother( size_t i = 0 ) const override;
    /// number of source candidates 
    /// ( the candidates used to construct this Candidate). 
    /// for CompositeRefBaseCandidates, the source candidates 
    /// are the daughters. 
    size_type numberOfSourceCandidatePtrs() const override ;
    /// return a RefToBase to one of the source Candidates 
    /// ( the candidates used to construct this Candidate). 
    /// for CompositeRefBaseCandidates, the source candidates 
    /// are the daughters. 
    CandidatePtr sourceCandidatePtr( size_type i ) const override;

  private:
    /// collection of references to daughters
    daughters dau;
    /// check overlap with another candidate
    bool overlap( const Candidate & ) const override;
  };

  inline void CompositePtrCandidate::addDaughter( const CandidatePtr & cand ) { 
    dau.push_back( cand ); 
  }

}

#endif
