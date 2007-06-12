#ifndef Candidate_CompositeCandidate_H
#define Candidate_CompositeCandidate_H
#include "DataFormats/Candidate/interface/Candidate.h"
#include <memory>
/** \class reco::CompositeCandidate
 *
 * A Candidate composed of daughters. 
 * The daughters are owned by the composite candidate.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CompositeCandidate.h,v 1.15 2007/05/14 11:59:26 llista Exp $
 *
 */

#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

namespace reco {

  class CompositeCandidate : public Candidate {
  public:
    /// collection of daughters
    typedef CandidateCollection daughters;
    /// default constructor
    CompositeCandidate() : Candidate() { }
    /// constructor from values
    CompositeCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			int pdgId = 0, int status = 0, bool integerCharge = true ) :
      Candidate( q, p4, vtx, pdgId, status, integerCharge ) { }
     /// constructor from values
    CompositeCandidate( const Particle & p ) :
      Candidate( p ) { }
   /// destructor
    virtual ~CompositeCandidate();
    /// returns a clone of the candidate
    virtual CompositeCandidate * clone() const;
    /// first daughter const_iterator
    virtual const_iterator begin() const;
    /// last daughter const_iterator
    virtual const_iterator end() const;
    /// first daughter iterator
    virtual iterator begin();
    /// last daughter const_iterator
    virtual iterator end();
    /// number of daughters
    virtual size_t numberOfDaughters() const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate * daughter( size_type ) const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate * daughter( size_type );
    /// add a clone of the passed candidate as daughter 
    void addDaughter( const Candidate & );
    /// add a clone of the passed candidate as daughter 
    void addDaughter( std::auto_ptr<Candidate> );
    /// clear daughters
    void clearDaughters() { dau.clear(); }
  private:
    // const iterator implementation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    // iterator implementation
    typedef candidate::iterator_imp_specific<daughters> iterator_imp_specific;
    /// collection of daughters
    daughters dau;
    /// check overlap with another daughter
    virtual bool overlap( const Candidate & ) const;
    /// post-read fixup
    virtual void fixup() const;
  };

  inline void CompositeCandidate::addDaughter( const Candidate & cand ) { 
    Candidate * c = cand.clone();
    dau.push_back( c ); 
  }

  inline void CompositeCandidate::addDaughter( std::auto_ptr<Candidate> cand ) {
    dau.push_back( cand );
  }

}

#endif
