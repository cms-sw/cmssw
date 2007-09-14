#ifndef Candidate_CompositeRefBaseCandidate_h
#define Candidate_CompositeRefBaseCandidate_h
#include "DataFormats/Candidate/interface/Candidate.h"
/** \class reco::CompositeRefBaseCandidate
 *
 * a reco::Candidate composed of daughters. 
 * The daughters has persistent references (edm::RefToBase<...>) 
 * to reco::Candidate stored in a separate collection.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CompositeRefBaseCandidate.h,v 1.11 2007/06/12 21:27:21 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

namespace reco {

  class CompositeRefBaseCandidate : public Candidate {
  public:
    /// collection of references to daughters
    typedef std::vector<CandidateBaseRef> daughters;
    /// default constructor
    CompositeRefBaseCandidate() : Candidate() { }
    /// constructor from values
    CompositeRefBaseCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			       int pdgId = 0, int status = 0, bool integerCharge = true ) :
      Candidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// constructor from a particle
    CompositeRefBaseCandidate( const Particle & p ) : Candidate( p ) { }
    /// destructor
    virtual ~CompositeRefBaseCandidate();
    /// returns a clone of the candidate
    virtual CompositeRefBaseCandidate * clone() const;
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
    void addDaughter( const CandidateBaseRef & );    
    /// clear daughter references
    void clearDaughters() { dau.clear(); }
    /// reference to daughter at given position
    CandidateBaseRef daughterRef( size_type i ) const { return dau[ i ]; }

  private:
    /// const iterator implentation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    /// iterator implentation
    typedef candidate::iterator_imp_specific_dummy<daughters> iterator_imp_specific;
    /// collection of references to daughters
    daughters dau;
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// post-read fixup operation
    /// warning: no way to automatically set mother references
    /// because no unique ProductID is stored here.
    /// Mother links will not be automatically set up
    /// for this class.
    virtual void doFixupMothers() const;
  };

  inline void CompositeRefBaseCandidate::addDaughter( const CandidateBaseRef & cand ) { 
    dau.push_back( cand ); 
  }
}

#endif
