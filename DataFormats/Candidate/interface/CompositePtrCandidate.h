#ifndef Candidate_CompositePtrCandidate_h
#define Candidate_CompositePtrCandidate_h
#include "DataFormats/Candidate/interface/Candidate.h"
/** \class reco::CompositePtrCandidate
 *
 * a reco::Candidate composed of daughters. 
 * The daughters has persistent references (edm::Ptr <...>) 
 * to reco::Candidate stored in a separate collection.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CompositePtrCandidate.h,v 1.2 2007/12/10 12:16:40 llista Exp $
 *
 */

#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

namespace reco {

  class CompositePtrCandidate : public Candidate {
  public:
    /// collection of references to daughters
    typedef std::vector<CandidatePtr> daughters;
    /// collection of references to daughters
    typedef std::vector<CandidatePtr> mothers;
    /// default constructor
    CompositePtrCandidate() : Candidate() { }
    /// constructor from values
    CompositePtrCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      Candidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from values
    CompositePtrCandidate( Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      Candidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from a particle
    explicit CompositePtrCandidate( const Particle & p ) : Candidate( p ) { }
    /// destructor
    virtual ~CompositePtrCandidate();
    /// returns a clone of the candidate
    virtual CompositePtrCandidate * clone() const;
    /// first daughter const_iterator
    virtual const_iterator begin() const;
    /// last daughter const_iterator
    virtual const_iterator end() const;
    /// first daughter iterator
    virtual iterator begin();
    /// last daughter iterator
    virtual iterator end();
    /// number of daughters
    virtual size_t numberOfDaughters() const;
    /// number of mothers
    virtual size_t numberOfMothers() const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate * daughter( size_type ) const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate * daughter( size_type );
    /// add a daughter via a reference
    void addDaughter( const CandidatePtr & );    
    /// clear daughter references
    void clearDaughters() { dau.clear(); }
    /// reference to daughter at given position
    CandidatePtr daughterPtr( size_type i ) const { return dau[ i ]; }
    /// references to daughtes
    const daughters & daughterPtrVector() const { return dau; }
    /// return pointer to mother
    virtual const Candidate * mother( size_t i = 0 ) const;

  private:
    /// const iterator implementation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    /// iterator implementation
    typedef candidate::iterator_imp_specific_dummy<daughters> iterator_imp_specific;
    /// collection of references to daughters
    daughters dau;
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
  };

  inline void CompositePtrCandidate::addDaughter( const CandidatePtr & cand ) { 
    dau.push_back( cand ); 
  }

}

#endif
