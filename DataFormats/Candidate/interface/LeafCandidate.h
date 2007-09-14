#ifndef Candidate_LeafCandidate_h
#define Candidate_LeafCandidate_h
/** \class reco::LeafCandidate
 *
 * particle candidate with no constituent nor daughters
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: LeafCandidate.h,v 1.12 2007/06/12 21:27:21 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

namespace reco {
  
  class LeafCandidate : public Candidate {
  public:
    /// collection of daughter candidates
    typedef CandidateCollection daughters;
    /// default constructor
    LeafCandidate() : Candidate() { }
    /// constructor from Particle
    explicit LeafCandidate( const Particle & p ) : Candidate( p ) { }
    /// constructor from values
    LeafCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
		   int pdgId = 0, int status = 0, bool integerCharge = true ) : 
      Candidate( q, p4, vtx, pdgId, status, integerCharge ) { }
    /// destructor
    virtual ~LeafCandidate();
    /// returns a clone of the Candidate object
    virtual LeafCandidate * clone() const;
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
    /// return daughter at a given position (throws an exception)
    virtual const Candidate * daughter( size_type ) const;
    /// return daughter at a given position (throws an exception)
    virtual Candidate * daughter( size_type );

  private:
    // const iterator implementation
    typedef candidate::const_iterator_imp_specific<daughters> const_iterator_imp_specific;
    // iterator implementation
    typedef candidate::iterator_imp_specific<daughters> iterator_imp_specific;
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & c ) const;
    /// post-read fixup operation
    virtual void doFixupMothers() const;
  };

}

#endif
