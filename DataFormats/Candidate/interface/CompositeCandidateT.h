#ifndef Candidate_CompositeCandidateT_h
#define Candidate_CompositeCandidateT_h
#include "DataFormats/Candidate/interface/Candidate.h"
#include <memory>
/** \class reco::CompositeCandidateT<D>
 *
 * A Candidate composed of daughters. 
 * The daughters are owned by the composite candidate.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CompositeCandidateT.h,v 1.16 2007/06/12 21:27:21 llista Exp $
 *
 */

#include "DataFormats/Candidate/interface/iterator_imp_specific.h"

namespace reco {

  template<typename D>
  class CompositeCandidateT : public Candidate {
  public:
    /// default constructor
    CompositeCandidateT() : Candidate() { }
    /// constructor from values
    CompositeCandidateT( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ),
			int pdgId = 0, int status = 0, bool integerCharge = true ) :
      Candidate( q, p4, vtx, pdgId, status, integerCharge ) { }
     /// constructor from values
    CompositeCandidateT( const Particle & p ) :
      Candidate( p ) { }
   /// destructor
    virtual ~CompositeCandidateT();
    /// returns a clone of the candidate
    virtual CompositeCandidateT * clone() const;
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
    typedef candidate::const_iterator_imp_specific<D> const_iterator_imp_specific;
    // iterator implementation
    typedef candidate::iterator_imp_specific<D> iterator_imp_specific;
    /// collection of daughters
    D dau;
    /// check overlap with another daughter
    virtual bool overlap( const Candidate & ) const;
    /// post-read fixup
    virtual void fixup() const;
  };

  template<typename D>
  inline void CompositeCandidateT<D>::addDaughter( const Candidate & cand ) { 
    Candidate * c = cand.clone();
    dau.push_back( c ); 
  }

  template<typename D>
  inline void CompositeCandidateT<D>::addDaughter( std::auto_ptr<Candidate> cand ) {
    dau.push_back( cand );
  }

  template<typename D>
  CompositeCandidateT<D>::~CompositeCandidateT() { }
  
  template<typename D>
  CompositeCandidateT * CompositeCandidateT<D>::clone() const { return new CompositeCandidateT( * this ); }
  
  template<typename D>
  Candidate::const_iterator CompositeCandidateT<D>::begin() const { return const_iterator( new const_iterator_imp_specific( dau.begin() ) ); }
  
  template<typename D>
  Candidate::const_iterator CompositeCandidateT<D>::end() const { return const_iterator( new const_iterator_imp_specific( dau.end() ) ); }    
  
  template<typename D>
  Candidate::iterator CompositeCandidateT<D>::begin() { return iterator( new iterator_imp_specific( dau.begin() ) ); }
  
  template<typename D>
  Candidate::iterator CompositeCandidateT<D>::end() { return iterator( new iterator_imp_specific( dau.end() ) ); }    
  
  template<typename D>
  const Candidate * CompositeCandidateT<D>::daughter( size_type i ) const { 
    return ( i >= 0 && i < numberOfDaughters() ) ? & dau[ i ] : 0;
  }
  
  template<typename D>
  Candidate * CompositeCandidateT<D>::daughter( size_type i ) { 
    return ( i >= 0 && i < numberOfDaughters() ) ? & dau[ i ] : 0;
  }
  
   template<typename D>
   size_t CompositeCandidateT<D>::numberOfDaughters() const { return dau.size(); }
  
  template<typename D>
  bool CompositeCandidateT<D>::overlap( const Candidate & c2 ) const {
    throw cms::Exception( "Error" ) << "can't check overlap internally for CompositeCanddate";
  }
  
  template<typename D>
  void CompositeCandidateT<D>::fixup() const {
    size_t n = numberOfDaughters();
    for( size_t i = 0; i < n; ++ i ) {
      daughter( i )->addMother( this );
    }
  }

}

#endif
