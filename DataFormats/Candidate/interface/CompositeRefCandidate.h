#ifndef Candidate_CompositeRefCandidate_h
#define Candidate_CompositeRefCandidate_h
#include "DataFormats/Candidate/interface/Candidate.h"
/** \class reco::CompositeRefCandidate
 *
 * a reco::Candidate composed of daughters. 
 * The daughters has persistent references (edm::Ref <...>) 
 * to reco::Candidate stored in a separate collection.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CompositeRefCandidate.h,v 1.13 2007/05/14 11:59:26 llista Exp $
 *
 */

#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

namespace reco {

  class CompositeRefCandidate : public Candidate {
  public:
    /// collection of references to daughters
    typedef CandidateRefVector daughters;
    /// default constructor
    CompositeRefCandidate() : Candidate() { }
    /// constructor from values
    CompositeRefCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			   int pdgId = 0, int status = 0, bool integerCharge = true ) :
      Candidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from a particle
    CompositeRefCandidate( const Particle & p ) : Candidate( p ) { }
    /// destructor
    virtual ~CompositeRefCandidate();
    /// returns a clone of the candidate
    virtual CompositeRefCandidate * clone() const;
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
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate * daughter( size_type ) const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate * daughter( size_type );
    /// add a daughter via a reference
    void addDaughter( const CandidateRef & );    
    /// clear daughter references
    void clearDaughters() { dau.clear(); }
    /// reference to daughter at given position
    CandidateRef daughterRef( size_type i ) const { return dau[ i ]; }
  private:
    /// const iterator implementation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    /// iterator implementation
    typedef candidate::iterator_imp_specific_dummy<daughters> iterator_imp_specific;
    /// collection of references to daughters
    daughters dau;
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// post-read fixup operation
    virtual void fixup() const;
  };

  inline void CompositeRefCandidate::addDaughter( const CandidateRef & cand ) { 
    dau.push_back( cand ); 
  }
}

#endif
